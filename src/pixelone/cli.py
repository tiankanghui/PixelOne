"""Command-line entry points for inference, training, data, caches, and export."""

import argparse
import json


def main():
    """Dispatch the five supported workflows without importing model code for help."""
    parser = argparse.ArgumentParser(prog='pixelone')
    commands = parser.add_subparsers(dest='command', required=True)
    inference = commands.add_parser('infer', help='Predict an original-image click')
    inference.add_argument('--model', required=True)
    inference.add_argument('--image', required=True)
    inference.add_argument('--instruction', required=True)
    inference.add_argument('--device', default='cuda:0')
    inference.add_argument('--revision')
    inference.add_argument(
        '--subfolder', help='Model directory within a Hugging Face collection or local path'
    )
    inference.add_argument('--kernel-backend', choices=['native', 'torch'])
    training = commands.add_parser('train', help='Train a search teacher or a direct student')
    training.add_argument('--config', required=True)
    training.add_argument('--resume')
    data = commands.add_parser('prepare-data', help='Validate original-pixel box annotations')
    data.add_argument('--input', required=True)
    data.add_argument('--output', required=True)
    data.add_argument('--image-root')
    cache = commands.add_parser('cache-teacher', help='Run T3 and cache teacher distributions')
    cache.add_argument('--config', required=True)
    cache.add_argument('--adapter', required=True)
    cache.add_argument('--output', required=True)
    export = commands.add_parser(
        'export', help='Export a full student checkpoint without optimizer state'
    )
    export.add_argument('--checkpoint', required=True)
    export.add_argument('--output', required=True)
    export.add_argument(
        '--precision', choices=['training_cuda', 'float32'], default='training_cuda'
    )
    arguments = parser.parse_args()
    if arguments.command == 'infer':
        from PIL import Image
        from pixelone.inference import PixelOnePredictor

        predictor = PixelOnePredictor.from_pretrained(
            arguments.model,
            device=arguments.device,
            revision=arguments.revision,
            subfolder=arguments.subfolder,
            kernel_backend=arguments.kernel_backend,
        )
        with Image.open(arguments.image) as image:
            result = predictor.predict(image, arguments.instruction)
        print(json.dumps(result, ensure_ascii=False))
    elif arguments.command == 'train':
        from pathlib import Path
        from pixelone.training import run_training

        result = run_training(
            json.loads(Path(arguments.config).read_text()), resume=arguments.resume
        )
        print(json.dumps(result))
    elif arguments.command == 'prepare-data':
        from pixelone.data import prepare_dataset

        print(json.dumps(prepare_dataset(arguments.input, arguments.output, arguments.image_root)))
    elif arguments.command == 'export':
        from pixelone.export import export_checkpoint

        print(
            json.dumps(
                export_checkpoint(
                    arguments.checkpoint, arguments.output, precision=arguments.precision
                )
            )
        )
    elif arguments.command == 'cache-teacher':
        from pathlib import Path
        import torch
        from safetensors.torch import load_file
        from pixelone.training import load_native
        from pixelone.data import GroundingDataset
        from pixelone.teacher import SearchTeacher, cache_teacher
        from pixelone.model import Teacher
        from pixelone.runtime import native_runtime

        config = json.loads(Path(arguments.config).read_text())
        recipe = config['recipe']
        device = torch.device(config.get('device', 'cuda:0'))
        with native_runtime(recipe['kernel_backend']):
            native, processor = load_native(config['base_model'], device)
            model = Teacher(native, recipe)
            try:
                model.load_adapter(load_file(arguments.adapter, device='cpu'))
                cache_teacher(
                    SearchTeacher(model, processor, recipe),
                    GroundingDataset(config['data'], config.get('image_root')),
                    arguments.output,
                )
            finally:
                model.close()


if __name__ == '__main__':
    main()
