import argparse
import io
import json
from pathlib import Path
import zipfile

import torch

from upar.inference import load_checkpoint_model


INFERENCE_MODULES = ('__init__.py', 'attributes.py', 'config.py', 'model.py', 'vit.py', 'e1.py', 'irra.py', 'clip_text.py', 'data.py', 'retrieval.py', 'inference.py')


def quantize_int8(state, min_size=65536, group=64):
    """Large weight matrices (names ending in 'weight', at least `min_size` values) as int8 with a
    float16 scale per output row and group of `group` inputs (symmetric, absmax / 127; one scale
    per row when the row length is not a multiple of `group`); everything else is unchanged.
    upar.inference.load_checkpoint_model restores them from the scale shape."""
    quantized, scales = {}, {}
    for key, value in state.items():
        if key.endswith('weight') and value.is_floating_point() and value.ndim >= 2 and value.numel() >= min_size:
            rows = value.float().reshape(value.shape[0], -1)
            width = group if rows.shape[1] % group == 0 else rows.shape[1]
            blocks = rows.reshape(rows.shape[0], -1, width)
            scale = (blocks.abs().amax(2, keepdim=True) / 127).clamp(min=1e-12)
            quantized[key] = (blocks / scale).round().clamp(-127, 127).to(torch.int8).reshape(value.shape)
            scales[key] = scale.half()  # [rows, groups, 1]
        else:
            quantized[key] = value
    return quantized, scales


def export_artifact(path, args, int8=False):
    _, checkpoint = load_checkpoint_model(path, 'cpu')
    if checkpoint.get('training_config', {}).get('holdout_domain'):
        raise ValueError(f'{path}: leave-one-domain-out checkpoints are validation-only; train on all domains to submit')
    # Only inference fields travel in the submission ZIP.
    artifact = {key: checkpoint[key] for key in (
        'format_version', 'architecture', 'attribute_names', 'preprocess', 'model_state',
        'inference_batch_size', 'query_chunk_size', 'epoch', 'validation_mADM', 'training_config')}
    artifact['weight_source'] = checkpoint.get('weight_source', 'model')
    if 'irra_bpe' in checkpoint:  # vocabulary for the IRRA text tower
        artifact['irra_bpe'] = checkpoint['irra_bpe']
    if int8:
        artifact['model_state'] = {k: v.half() if v.is_floating_point() else v for k, v in artifact['model_state'].items()}
    # E1 scores at temperature 0.01, which would amplify weight rounding: it stays float16.
    if int8 and artifact['architecture'] != 'e1_clip_l14':
        artifact['model_state'], artifact['int8_scales'] = quantize_int8(artifact['model_state'])
    if args.batch_size is not None:
        artifact['inference_batch_size'] = args.batch_size
    if args.retrieval is not None:
        # Inference reads the distance from training_config; the original value is kept for provenance.
        trained_with = artifact['training_config'].get('retrieval', 'l1')
        artifact['training_config'] = {**artifact['training_config'], 'retrieval': args.retrieval,
                                       'retrieval_trained_with': trained_with}
    buffer = io.BytesIO(); torch.save(artifact, buffer)
    return buffer.getvalue(), artifact


def main():
    parser = argparse.ArgumentParser(description='Build offline Track 2 ZIP with run.py at archive root')
    parser.add_argument('--checkpoint', required=True, nargs='+',
                        help='Inference checkpoint(s); several are averaged as an ensemble')
    parser.add_argument('--output', default='outputs/submission.zip')
    parser.add_argument('--batch-size', type=int, help='Override inference batch size for submission hardware')
    parser.add_argument('--retrieval', choices=('l1', 'bce'), help='Override the checkpoint retrieval distance')
    parser.add_argument('--int8-weights', action='store_true',
                        help='Store large weight matrices as int8 with per-row scales (rest float16; about a quarter '
                             'of the float32 size); restored to float at load time')
    parser.add_argument('--unsmooth', type=float, default=0.0,
                        help='Undo this much label smoothing per member before averaging (members trained '
                             'with label smoothing only), p\' = (p - eps/2) / (1 - eps); choose on LODO')
    parser.add_argument('--mixture', type=float, nargs=2, metavar=('T', 'W'),
                        help='Complete-query likelihood mixture: (1 - W) * BCE + W * D_T (BCE retrieval)')
    parser.add_argument('--irra-text', type=float, default=0.0, metavar='LAMBDA',
                        help='Subtract LAMBDA * cos(IRRA image, query prompt) from the distances (needs an IRRA '
                             'member converted with --text); choose on LODO')
    parser.add_argument('--irra-text-only', action='store_true',
                        help='With --irra-text, use the IRRA member only for the image-text score: its attribute '
                             'probabilities are left out of the ensemble average and the mixture')
    parser.add_argument('--smooth', type=float, nargs=2, metavar=('K', 'W'),
                        help='Gallery neighbour smoothing of the final distances: (1 - W) * D + W * mean of D over '
                             'the K nearest gallery images by IRRA image cosine (needs --irra-text); choose on LODO')
    parser.add_argument('--smooth-features', type=int, metavar='INDEX',
                        help='With --smooth, find the neighbours with this member\'s attribute-head input instead of '
                             'the IRRA image embeddings (checkpoint index)')
    parser.add_argument('--smooth-mutual', action='store_true',
                        help='With --smooth, average only over mutual neighbours (images without any stay unchanged)')
    parser.add_argument('--amp-inference', action='store_true',
                        help='Run forward passes under float16 autocast on CUDA (about 2x faster on Tensor Core GPUs)')
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('batch-size must be positive')
    torch.set_num_threads(4)
    exported = [export_artifact(path, args, args.int8_weights) for path in args.checkpoint]
    retrieval = {a['training_config'].get('retrieval', 'l1') for _, a in exported}
    if len(retrieval) != 1:
        parser.error(f'Ensemble members use different retrieval distances {sorted(retrieval)}; pass --retrieval')
    if not 0 <= args.unsmooth < 1:
        parser.error('--unsmooth must be in [0, 1)')
    unsmooth = [args.unsmooth if a['training_config'].get('label_smoothing', 0) > 0 else 0.0 for _, a in exported]
    irra_text = None
    if args.irra_text:
        index = [i for i, (_, a) in enumerate(exported) if 'irra_bpe' in a]
        if len(index) != 1:
            parser.error('--irra-text needs exactly one IRRA member converted with --text')
        irra_text = {'member': index[0], 'lambda': args.irra_text}
        if args.irra_text_only:
            if len(exported) < 2:
                parser.error('--irra-text-only needs at least one other member')
            irra_text['vote'] = False
    elif args.irra_text_only:
        parser.error('--irra-text-only needs --irra-text')
    smooth = None
    if args.smooth:
        k, w = args.smooth
        if not irra_text or k < 1 or k != int(k) or not 0 < w <= 1:
            parser.error('--smooth needs --irra-text, an integer K >= 1 and 0 < W <= 1')
        smooth = {'k': int(k), 'weight': w, 'mutual': args.smooth_mutual}
        if args.smooth_features is not None:
            if not 0 <= args.smooth_features < len(exported) or 'irra_bpe' in exported[args.smooth_features][1]:
                parser.error('--smooth-features must index a non-IRRA checkpoint')
            smooth['member'] = args.smooth_features
    elif args.smooth_mutual or args.smooth_features is not None:
        parser.error('--smooth-mutual and --smooth-features need --smooth')
    mixture = None
    if args.mixture:
        temperature, weight = args.mixture
        if temperature <= 0 or not 0 <= weight <= 1 or retrieval != {'bce'}:
            parser.error('--mixture needs T > 0, 0 <= W <= 1 and BCE retrieval')
        mixture = {'temperature': temperature, 'weight': weight}
    root = Path(__file__).resolve().parent
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f'{target} already exists; choose a new output path')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.write(root / 'submission/run.py', 'run.py')
        # Preserve organizer metadata byte-for-byte.
        archive.write(root / 'examples/task2/sample_code_submission/metadata.yaml', 'metadata.yaml')
        for name in INFERENCE_MODULES:
            archive.write(root / 'upar' / name, 'upar/' + name)
        if len(exported) == 1 and not (args.amp_inference or any(unsmooth) or mixture or irra_text or smooth):
            archive.writestr('assets/model.pt', exported[0][0])
        else:
            names = [f'model_{i}.pt' for i in range(len(exported))]
            for name, (data, _) in zip(names, exported):
                archive.writestr('assets/' + name, data)
            manifest = {'models': names, 'amp': args.amp_inference, 'retrieval': retrieval.pop(), 'unsmooth': unsmooth,
                        'mixture': mixture, 'irra_text': irra_text, 'smooth': smooth,
                        'sources': [str(Path(p)) for p in args.checkpoint]}
            archive.writestr('assets/ensemble.json', json.dumps(manifest, indent=2))
    print(f'Created {target}: {target.stat().st_size / 2**20:.1f} MiB, {len(exported)} model(s), '
          f'int8={args.int8_weights}, amp_inference={args.amp_inference}, unsmooth={unsmooth}, mixture={mixture}, '
          f'irra_text={irra_text}, smooth={smooth}')
    print('Inference imports only the standard library, torch, torchvision, numpy and Pillow.')


if __name__ == '__main__':
    main()
