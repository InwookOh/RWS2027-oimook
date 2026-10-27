import argparse
import json
import time
from pathlib import Path

import torch

from upar.results import save_predictions, load_predictions, append_results
from upar.data import load_split
from upar.evaluation import evaluate_split
from upar.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description='Evaluate official validation retrieval metrics')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', nargs='+', help='One checkpoint, or several averaged as an ensemble')
    source.add_argument('--cache', help='Evaluate saved predictions without model/image inference')
    parser.add_argument('--retrieval', choices=('l1', 'bce'))
    parser.add_argument('--tta-flip', action='store_true', help='Average original and flipped-image predictions')
    parser.add_argument('--amp', action='store_true', help='float16 autocast forward passes (CUDA)')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs/evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--max-images', type=int, help='Smoke test only; first N validation rows')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if args.cache:
        if args.max_images is not None:
            parser.error('--max-images is only supported with --checkpoint')
        cached = load_predictions(args.cache)
        names = cached['image_names'].tolist()
        labels, queries, ids, probabilities = (cached[k] for k in ('labels', 'queries', 'ids', 'probs'))
        metadata = cached['metadata']
        retrieval = args.retrieval or metadata.get('config', {}).get('retrieval', 'l1')
        annotations_finished = model_finished = gallery_finished = time.monotonic()
        cache_path = Path(args.cache).resolve()
        runtime = {'device': 'cpu', 'prediction_cache': True, 'cpu_threads': torch.get_num_threads()}
    else:
        names, labels, queries, ids = load_split(args.data_dir, 'val', args.max_images)
        annotations_finished = time.monotonic()
        engine = InferenceEngine(args.checkpoint, args.device, args.batch_size, args.workers, tta_flip=args.tta_flip,
                                 amp=args.amp)
        model_finished = time.monotonic()
        root = Path(args.data_dir).resolve()
        probabilities = engine.predict([str(root / p) for p in names])
        gallery_finished = time.monotonic()
        checkpoint = torch.load(args.checkpoint[0], map_location='cpu', weights_only=True)
        metadata = {'checkpoint': [str(Path(p).resolve()) for p in args.checkpoint], 'tta_flip': args.tta_flip, 'amp_inference': args.amp,
                    'epoch': checkpoint.get('epoch'), 'config': checkpoint.get('training_config', {}),
                    'subset_run': args.max_images is not None}
        retrieval = args.retrieval or engine.retrieval
        cache_path = output / 'val_predictions.npz'
        save_predictions(cache_path, probabilities, labels, queries, ids, names, metadata)
        runtime = {'device': str(engine.device), 'torch': str(torch.__version__),
                   'batch_size': engine.batch_size, 'workers': engine.workers,
                   'cpu_threads': torch.get_num_threads()}
    report = evaluate_split(queries, labels, ids, probabilities, names, retrieval=retrieval)
    metrics_finished = time.monotonic()
    report['checkpoint'] = metadata.get('checkpoint')
    report['cache'] = str(cache_path)
    report['subset_run'] = metadata.get('subset_run', False)
    report['retrieval'] = retrieval
    report['timings'] = {'annotation_load_seconds': annotations_finished - started,
                         'model_load_seconds': model_finished - annotations_finished,
                         'gallery_inference_seconds': gallery_finished - model_finished,
                         'metrics_seconds': metrics_finished - gallery_finished,
                         'total_through_metrics_seconds': metrics_finished - started}
    report['runtime'] = runtime
    append_results(output / 'results.csv', report,
                   {'run': str(output.resolve()), 'epoch': metadata.get('epoch'),
                    'seed': metadata.get('config', {}).get('seed', metadata.get('seed')),
                    'amp': metadata.get('amp', metadata.get('config', {}).get('amp', False)),
                    'retrieval': retrieval, 'subset_run': report['subset_run'], 'cache': str(cache_path),
                    'gallery_inference_seconds': 0 if args.cache else gallery_finished - model_finished,
                    'metrics_seconds': metrics_finished - gallery_finished})
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
