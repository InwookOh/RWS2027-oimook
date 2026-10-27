from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .data import ImageDataset, make_transform


def binary_matrix(values, name):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 40:
        raise ValueError(f'{name} must have shape (N, 40)')
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f'{name} must contain only 0/1')
    return array


def l1_distances(queries, probabilities):
    """Expected attribute mismatches; no Q x G x 40 intermediate."""
    queries = binary_matrix(queries, 'queries')
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != 40:
        raise ValueError('Probabilities must have shape (G, 40)')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    distance = probabilities.sum(1)[None, :] + queries @ (1 - 2 * probabilities).T
    # Floating point cancellation can produce tiny negative distances.
    return np.clip(distance, 0, 40)


def bce_distances(queries, probabilities):
    """Negative log likelihood of complete binary descriptions; smaller is better."""
    queries = binary_matrix(queries, 'queries')
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != 40:
        raise ValueError('Probabilities must have shape (G, 40)')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    absent = np.log1p(-p)
    return np.maximum(0, -absent.sum(1)[None, :] + queries @ (absent - np.log(p)).T)


DISTANCE_FUNCTIONS = {'l1': l1_distances, 'bce': bce_distances}
RETRIEVAL_METHODS = tuple(DISTANCE_FUNCTIONS)


def check_retrieval(method):
    """Validate a retrieval name (argparse type)."""
    if method not in DISTANCE_FUNCTIONS:
        raise ValueError(f'Unsupported retrieval method: {method}')
    return method


def query_distances(queries, probabilities, method='l1'):
    return DISTANCE_FUNCTIONS[check_retrieval(method)](queries, probabilities)


@torch.inference_mode()
def predict_probabilities(model, paths, preprocess, device, batch_size=32, workers=0, flip=False, amp=False):
    return predict_ensemble([(model, preprocess)], paths, device, batch_size, workers, flip, amp)


@torch.inference_mode()
def predict_ensemble(members, paths, device, batch_size=32, workers=0, flip=False, amp=False, unsmooth=None,
                     return_members=False, threads=8, prefetch=4, serial=False, globals_of=()):
    """Mean sigmoid probabilities of (model, preprocess) members.

    Images are normalized per member on the device, so members sharing an input size share
    the decoded batch. With workers = 0 (submissions) each image is decoded once by a pool of
    `threads` threads and resized for every input size; with workers > 0 a DataLoader decodes
    once per input size; `serial` keeps that one-pass-per-size loader in the main process. `flip` is one bool for
    all members or one per member (horizontal-flip TTA). With `amp` on CUDA, forward passes
    run under float16 autocast (as in training); sigmoids stay float32. `unsmooth` gives one
    epsilon per member (0 = off): p' = clip((p - eps / 2) / (1 - eps)) undoes part of label
    smoothing before averaging. With `return_members`, the per-member probabilities [M, N, 40]
    are returned too. `globals_of` lists members whose global image embeddings are collected too:
    IRRA's own (`keep_global`), for other members the input of the attribute head (`model.head`);
    then a third value maps member index -> [N, D] float32.
    """
    if batch_size < 1 or workers < 0 or not members:
        raise ValueError('Invalid batch size/worker count or empty ensemble')
    flips = [bool(f) for f in flip] if isinstance(flip, (list, tuple)) else [bool(flip)] * len(members)
    if len(flips) != len(members):
        raise ValueError('Need one flip setting per ensemble member')
    unsmooth = [0.0] * len(members) if unsmooth is None else [float(e) for e in unsmooth]
    if len(unsmooth) != len(members) or not all(0 <= e < 1 for e in unsmooth):
        raise ValueError('Need one unsmoothing epsilon in [0, 1) per ensemble member')
    per_member = np.zeros((len(members), len(paths), 40), dtype=np.float32) if return_members else None
    embeddings = {}
    for index in globals_of:
        model = members[index][0]
        if hasattr(model, 'keep_global'):
            model.keep_global = True
        elif getattr(model, 'global_hook', None) is None:
            def keep(module, inputs, owner=model):
                owner.last_global = inputs[0]
            model.global_hook = model.head.register_forward_pre_hook(keep)
    device = torch.device(device)
    use_amp = bool(amp and device.type == 'cuda')
    total = np.zeros((len(paths), 40), dtype=np.float64)
    sizes = {}
    for index, ((model, preprocess), member_flip) in enumerate(zip(members, flips)):
        key = (int(preprocess['height']), int(preprocess['width']), preprocess.get('interpolation', 'bilinear'))
        sizes.setdefault(key, []).append((model, preprocess, member_flip, index))
    groups = list(sizes.values())
    stats = [[(torch.tensor(p['mean'], device=device).view(1, 3, 1, 1),
               torch.tensor(p['std'], device=device).view(1, 3, 1, 1)) for _, p, _, _ in group] for group in groups]
    for group in groups:
        for model, _, _, _ in group:
            model.eval()

    def run(group, group_stats, images, offset):
        images = images.to(device, non_blocking=True)
        for (model, _, member_flip, index), (mean, std) in zip(group, group_stats):
            inputs = (images - mean) / std
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(inputs).float().sigmoid()
                if index in globals_of:
                    embedding = model.last_global.float().cpu().numpy()
                    if index not in embeddings:
                        embeddings[index] = np.zeros((len(paths), embedding.shape[1]), dtype=np.float32)
                    embeddings[index][offset:offset + len(images)] = embedding
                if member_flip:
                    # UPAR attributes have no left/right distinction, so labels are flip-invariant.
                    prediction = (prediction + model(inputs.flip(3)).float().sigmoid()) / 2
            if unsmooth[index]:
                eps = unsmooth[index]
                prediction = ((prediction - eps / 2) / (1 - eps)).clamp(1e-6, 1 - 1e-6)
            prediction = prediction.double().cpu().numpy()
            total[offset:offset + len(images)] += prediction
            if per_member is not None:
                per_member[index, offset:offset + len(images)] = prediction

    if workers > 0 or serial:
        # One pass per input size (training-time evaluation with worker processes, or the
        # earlier single-threaded submission loader with serial=True).
        for group, group_stats in zip(groups, stats):
            loader = DataLoader(
                ImageDataset(paths, make_transform(group[0][1], normalize=False)), batch_size=batch_size,
                shuffle=False, num_workers=workers, pin_memory=device.type == 'cuda',
            )
            offset = 0
            for step, images in enumerate(loader):
                run(group, group_stats, images, offset)
                offset += len(images)
                if step == 0 or (step + 1) % 100 == 0 or offset == len(paths):
                    print(f'Gallery inference: {offset}/{len(paths)} ({len(group)} model(s))', flush=True)
    else:
        # Submission path: each image is decoded once and resized for every input size by a
        # thread pool (no worker processes, so no shared-memory limits), a few batches ahead.
        # Threads only run PIL and return uint8 arrays; the float conversion (as ToTensor) runs
        # on the device, so no torch CPU ops compete across threads.
        resizes = [make_transform(group[0][1], normalize=False).transforms[0] for group in groups]

        def load(path):
            with Image.open(path) as im:
                im = im.convert('RGB')
                return [np.asarray(resize(im)) for resize in resizes]

        def to_device(arrays):
            batch = torch.from_numpy(np.stack(arrays)).to(device, non_blocking=True)
            return batch.permute(0, 3, 1, 2).float().div_(255)

        starts = iter(range(0, len(paths), batch_size))
        with ThreadPoolExecutor(threads) as pool:
            def submit():
                start = next(starts, None)
                if start is not None:
                    pending.append((start, [pool.submit(load, p) for p in paths[start:start + batch_size]]))

            pending = deque()
            for _ in range(prefetch):
                submit()
            step, started = 0, time.monotonic()
            while pending:
                offset, futures = pending.popleft()
                submit()
                items = [future.result() for future in futures]
                for g, (group, group_stats) in enumerate(zip(groups, stats)):
                    run(group, group_stats, to_device([item[g] for item in items]), offset)
                step += 1
                done = offset + len(items)
                # Few lines: the Codabench ingestion log keeps only its beginning, and the final
                # timing line must stay visible.
                if step == 1 or step % 300 == 0 or done == len(paths):
                    print(f'Gallery inference: {done}/{len(paths)} ({len(members)} model(s), '
                          f'{len(groups)} input size(s), {threads} threads, '
                          f'{time.monotonic() - started:.0f} s)', flush=True)
    probabilities = (total / len(members)).astype(np.float32)
    if not np.isfinite(probabilities).all():
        raise ValueError('Non-finite model predictions')
    if globals_of:
        return probabilities, per_member, embeddings
    return (probabilities, per_member) if return_members else probabilities


@torch.inference_mode()
def subtract_similarity(distances, text, images, weight, device='cpu', chunk_size=256):
    """distances -= weight * cos(text_q, image_g), in place (normalized embeddings)."""
    device = torch.device(device)
    images = torch.as_tensor(images, device=device)
    text = torch.as_tensor(text, device=device)
    for start in range(0, len(text), chunk_size):
        similarity = (text[start:start + chunk_size] @ images.T).cpu().numpy()
        distances[start:start + chunk_size] -= (weight * similarity).astype(distances.dtype)
    return distances


@torch.inference_mode()
def smooth_distances(distances, images, k, weight, device='cpu', chunk_size=2048, query_chunk=64, mutual=False):
    """Gallery neighbour smoothing, in place: D <- (1 - weight) * D + weight * mean of D over each
    image's k nearest other gallery images by cosine of `images` (IRRA image embeddings). With
    `mutual`, only mutual neighbours count: D <- D + weight / k * sum over them of (D_j - D)."""
    if k < 1 or not 0 <= weight <= 1 or k >= len(images):
        raise ValueError('Invalid neighbour count or smoothing weight')
    device = torch.device(device)
    x = torch.nn.functional.normalize(torch.as_tensor(images, device=device, dtype=torch.float32), dim=1)
    neighbours = torch.empty((len(x), k), dtype=torch.long, device=device)
    for start in range(0, len(x), chunk_size):
        sims = x[start:start + chunk_size] @ x.T
        rows = torch.arange(len(sims), device=device)
        sims[rows, start + rows] = -float('inf')
        neighbours[start:start + chunk_size] = sims.topk(k, dim=1).indices
    keep = None
    if mutual:
        own = torch.arange(len(x), device=device)[:, None, None]
        keep = (neighbours[neighbours] == own).any(-1).float()  # G x k
    for start in range(0, len(distances), query_chunk):
        d = torch.as_tensor(distances[start:start + query_chunk], device=device, dtype=torch.float32)
        if keep is None:
            smoothed = (1 - weight) * d + weight * d[:, neighbours].mean(-1)
        else:
            smoothed = d + weight / k * ((d[:, neighbours] - d[:, :, None]) * keep).sum(-1)
        distances[start:start + query_chunk] = smoothed.cpu().numpy().astype(distances.dtype)
    return distances


@torch.inference_mode()
def mixture_distance_matrix(queries, probabilities, members, chunk_size=64, temperature=4.0, weight=0.5,
                            device='cpu'):
    """(1 - weight) * BCE(averaged probabilities) + weight * D_T, where
    D_T = -T log mean_m exp(-BCE_m / T) mixes the members' complete-query likelihoods.
    Computed in chunks of queries on `device` (Q x G x members on the GPU is cheap)."""
    queries = binary_matrix(queries, 'queries')
    if chunk_size < 1 or temperature <= 0 or not 0 <= weight <= 1:
        raise ValueError('Invalid chunk size, temperature or mixture weight')
    device = torch.device(device)

    def logs(p):
        p = torch.as_tensor(np.asarray(p, dtype=np.float32), device=device).clamp(1e-6, 1 - 1e-6)
        absent = torch.log1p(-p)
        return -absent.sum(-1), absent - torch.log(p)  # [..., G], [..., G, 40]

    base_const, base_diff = logs(probabilities)
    member_const, member_diff = logs(members)  # [M, G], [M, G, 40]
    result = np.empty((len(queries), len(probabilities)), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        q = torch.as_tensor(queries[start:start + chunk_size], device=device)
        base = (base_const[None, :] + q @ base_diff.T).clamp(min=0)
        each = (member_const[:, None, :] + torch.einsum('qa,mga->mqg', q, member_diff)).clamp(min=0)
        low = each.min(0).values
        mixture = low - temperature * torch.log(torch.exp(-(each - low) / temperature).mean(0))
        result[start:start + chunk_size] = ((1 - weight) * base + weight * mixture).cpu().numpy()
    return result


def distance_matrix(queries, probabilities, chunk_size=64, method='l1'):
    queries = binary_matrix(queries, 'queries')
    if chunk_size < 1:
        raise ValueError('Query chunk size must be positive')
    result = np.empty((len(queries), len(probabilities)), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        result[start:start + chunk_size] = query_distances(queries[start:start + chunk_size], probabilities, method)
    return result
