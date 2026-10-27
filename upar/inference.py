from pathlib import Path
import json
import time

import numpy as np
import torch

from .attributes import ATTRIBUTES, validate_attribute_names
from .config import model_options
from .model import build_model
from .retrieval import distance_matrix, mixture_distance_matrix, predict_ensemble, smooth_distances, subtract_similarity


def load_checkpoint_model(path, device):
    # Exported artifacts contain only tensors and primitive Python values.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('attribute_names') != list(ATTRIBUTES):
        raise ValueError('Unsupported checkpoint or attribute order')
    model = build_model(checkpoint['architecture'], pretrained=False, **model_options(checkpoint.get('training_config', {})))
    # Weights may be stored in float16, or as int8 with per-row scales (build_submission.py
    # --int8-weights), to shrink the ZIP; inference always loads float32 weights.
    scales = checkpoint.get('int8_scales') or {}
    state = {}
    for key, value in checkpoint['model_state'].items():
        if key in scales:
            scale = scales[key].float()  # [rows, groups, 1] (or [rows, 1] for one scale per row)
            value = (value.float().reshape(value.shape[0], scale.shape[1] if scale.ndim == 3 else 1, -1)
                     * scale.reshape(value.shape[0], -1, 1)).reshape(value.shape)
        state[key] = value.float() if value.is_floating_point() else value
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def resolve_checkpoints(source):
    """Checkpoint paths and ensemble options from a file, a list of files, or an assets directory."""
    if isinstance(source, (list, tuple)):
        return [Path(p) for p in source], {}
    source = Path(source)
    if source.is_dir():
        manifest = source / 'ensemble.json'
        if manifest.exists():
            options = json.loads(manifest.read_text())
            return [source / name for name in options['models']], options
        return [source / 'model.pt'], {}
    return [source], {}


class InferenceEngine:
    """One model or an ensemble whose sigmoid probabilities are averaged, optionally with flip TTA."""

    def __init__(self, checkpoint_path, device=None, batch_size=None, workers=0, tta_flip=None, retrieval=None, amp=None):
        started = time.monotonic()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        paths, options = resolve_checkpoints(checkpoint_path)
        loaded = [load_checkpoint_model(path, self.device) for path in paths]
        print(f'Loaded {len(loaded)} model(s) in {time.monotonic() - started:.0f} s', flush=True)
        self.models = [model for model, _ in loaded]
        self.checkpoints = [checkpoint for _, checkpoint in loaded]
        self.preprocesses = [checkpoint['preprocess'] for _, checkpoint in loaded]
        checkpoint = loaded[0][1]
        self.model, self.preprocess = self.models[0], self.preprocesses[0]
        self.batch_size = batch_size or options.get('inference_batch_size') or checkpoint.get('inference_batch_size', 16)
        self.workers = workers
        self.chunk_size = checkpoint.get('query_chunk_size', 64)
        self.retrieval = retrieval or options.get('retrieval') or checkpoint.get('training_config', {}).get('retrieval', 'l1')
        # One bool for all members, or one per member (e.g. no flip for a heavy high-resolution member).
        flip = options.get('tta_flip', False) if tta_flip is None else tta_flip
        self.tta_flip = [bool(f) for f in flip] if isinstance(flip, (list, tuple)) else bool(flip)
        if isinstance(self.tta_flip, list) and len(self.tta_flip) != len(self.models):
            raise ValueError('tta_flip list must have one entry per model')
        # float16 autocast for the forward passes (CUDA only), as used in training.
        self.amp = bool(options.get('amp', False) if amp is None else amp)
        # Per-member label-smoothing correction and complete-query likelihood mixture (BCE only).
        self.unsmooth = options.get('unsmooth') or [0.0] * len(self.models)
        if len(self.unsmooth) != len(self.models):
            raise ValueError('unsmooth must have one entry per model')
        self.mixture = options.get('mixture')
        # 'threads' (decode once, thread pool) or 'serial' (one main-process pass per input size).
        self.loader = options.get('loader', 'threads')
        self.threads = int(options.get('threads', 8))
        if self.loader not in ('threads', 'serial'):
            raise ValueError(f'Unknown loader {self.loader!r}')
        if self.mixture and self.retrieval != 'bce':
            raise ValueError('mixture requires BCE retrieval')
        if options.get('query_norm'):
            raise ValueError('query-set normalization is no longer supported')
        # IRRA's CLIP image-text score: distance -= lambda * cos(text(query prompt), image).
        self.irra_text = options.get('irra_text')
        if self.irra_text and 'irra_bpe' not in self.checkpoints[self.irra_text['member']]:
            raise ValueError('irra_text needs an IRRA member converted with its text tower')
        # vote False: the IRRA member serves only the image-text score; its attribute
        # probabilities leave the ensemble average and the likelihood mixture.
        self.text_only = bool(self.irra_text and not self.irra_text.get('vote', True))
        # Gallery neighbour smoothing of the final distances (neighbours by IRRA image cosine).
        self.smooth = options.get('smooth')
        if self.smooth and not self.irra_text:
            raise ValueError('smooth needs irra_text (its image embeddings define the neighbours)')
        # Neighbours come from the IRRA member's image embeddings unless smooth names another member
        # (then that member's attribute-head input).
        self.smooth_member = int(self.smooth.get('member', self.irra_text['member'])) if self.smooth else None
        if self.smooth_member is not None and not 0 <= self.smooth_member < len(self.models):
            raise ValueError('smooth member must index a model')
        self.globals_of = tuple(sorted({self.irra_text['member'], self.smooth_member} - {None})) if self.irra_text else ()
        if self.text_only and len(self.models) < 2:
            raise ValueError('a text-only IRRA member needs at least one other member')

    def predict(self, paths, return_members=False, globals_of=()):
        return predict_ensemble(list(zip(self.models, self.preprocesses)), paths, self.device,
                                self.batch_size, self.workers, flip=self.tta_flip, amp=self.amp,
                                unsmooth=self.unsmooth, return_members=return_members,
                                serial=self.loader == 'serial', threads=self.threads, globals_of=globals_of)

    def rank_gallery(self, sample):
        started = time.monotonic()
        names = sample['attribute_names']
        validate_attribute_names(names)
        paths = [entry['image_path'] for entry in sample['gallery']]
        embeddings = None
        if self.irra_text:
            probabilities, members, embeddings = self.predict(paths, return_members=bool(self.mixture) or self.text_only,
                                                              globals_of=self.globals_of)
            if self.text_only:
                members = np.delete(members, self.irra_text['member'], axis=0)
                probabilities = members.mean(0)
        elif self.mixture:
            probabilities, members = self.predict(paths, return_members=True)
        else:
            probabilities, members = self.predict(paths), None
        gallery_finished = time.monotonic()
        # Output columns follow gallery list order, and probability columns follow
        # the supplied attribute names even if the ingestion ever reorders them.
        columns = [ATTRIBUTES.index(name) for name in names]
        probabilities = probabilities[:, columns]
        print(f'Gallery done in {gallery_finished - started:.0f} s; computing distances: '
              f'Q={len(sample["queries"])}, G={len(paths)}', flush=True)
        if self.mixture:
            result = mixture_distance_matrix(sample['queries'], probabilities, members[:, :, columns], self.chunk_size,
                                             float(self.mixture['temperature']), float(self.mixture['weight']),
                                             self.device)
        else:
            result = distance_matrix(sample['queries'], probabilities, self.chunk_size, self.retrieval)
        if self.irra_text:
            index = self.irra_text['member']
            queries = np.asarray(sample['queries'], dtype=np.int64)
            canonical = np.zeros((len(queries), len(ATTRIBUTES)), dtype=np.int64)
            canonical[:, columns] = queries  # the prompt template uses the canonical attribute order
            bpe = bytes(self.checkpoints[index]['irra_bpe'].tolist())
            text = self.models[index].query_features(canonical, bpe).cpu().numpy()
            subtract_similarity(result, text, embeddings[index], float(self.irra_text['lambda']), self.device)
            if self.smooth:
                smooth_distances(result, embeddings[self.smooth_member], int(self.smooth['k']), float(self.smooth['weight']),
                                 self.device, mutual=bool(self.smooth.get('mutual', False)))
        if not np.isfinite(result).all():
            raise ValueError('Non-finite distances')
        finished = time.monotonic()
        self.last_timings = {'gallery_inference_seconds': gallery_finished - started,
                             'retrieval_seconds': finished - gallery_finished,
                             'rank_gallery_seconds': finished - started,
                             'retrieval': self.retrieval, 'models': len(self.models), 'tta_flip': self.tta_flip,
                             'amp': self.amp and self.device.type == 'cuda',
                             'unsmooth': self.unsmooth, 'mixture': self.mixture, 'loader': self.loader,
                             'irra_text': self.irra_text, 'smooth': self.smooth}
        print('Inference timings: ' + json.dumps(self.last_timings), flush=True)
        return {'distances': result}
