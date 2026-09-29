#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX (GPU Runtime) and optionally TensorRT.

TensorRT engines are hardware-specific and should be built on the target GPU.
ONNX export does not require TensorRT. GPU ONNX inference uses CUDAExecutionProvider
when onnxruntime-gpu is installed.

Examples:
    python export_models.py --lock-dir ./results/cifar10/cifar10/locked
    python export_models.py --model_path ./locked/best.pt --model_name mobilenetv4 \\
        --num_classes 10 --onnx-only --device cuda
    python export_models.py --model_path ./locked/best.pt --model_name efficientnet_b0 \\
        --num_classes 4 --fp16 --device cuda
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms

from derms.config import MEAN, REGISTRY, STD
from derms.models import model_for
from derms.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Export trained models to ONNX and TensorRT.')
    parser.add_argument('--model_path', type=str, help='Path to the trained .pt checkpoint')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                        help='Registry alias (e.g. mobilenetv4) or timm identifier')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='exported_models')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu (ONNX GPU uses CUDA EP)')
    parser.add_argument('--fp16', action='store_true', help='TensorRT FP16')
    parser.add_argument('--int8', action='store_true', help='TensorRT INT8')
    parser.add_argument('--onnx-only', action='store_true', help='Skip TensorRT engine build')
    parser.add_argument('--lock-dir', type=Path,
                        help='locked/ directory; infers checkpoint, alias, and class count')
    parser.add_argument('--class_names', type=str, nargs='+',
                        help='Class names (space-separated). Inferred from lock.json when --lock-dir is set.')
    parser.add_argument('--test_image', type=str, default='test_image.jpg')
    return parser.parse_args()


def resolve_from_lock(lock_dir):
    lock_dir = Path(lock_dir)
    lock_path = lock_dir / 'lock.json'
    selection_path = lock_dir / 'selection.json'
    checkpoint = lock_dir / 'best.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f'No checkpoint at {checkpoint}')
    payload = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    selection = payload.get('selection') or (json.loads(selection_path.read_text()) if selection_path.exists() else {})
    winner = selection.get('winner', {})
    classes = payload.get('classes') or []
    alias = winner.get('model') or winner.get('architecture')
    if not alias:
        raise ValueError('lock.json/selection.json must contain winner.model')
    return {
        'model_path': str(checkpoint),
        'model_name': alias,
        'num_classes': len(classes) if classes else int(winner.get('number_of_folds') or 0) or None,
        'class_names': classes,
        'output_dir': str(lock_dir / 'export'),
    }


def state_dict_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Invalid checkpoint format.')
    for key in ('model', 'model_state_dict', 'state_dict'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise RuntimeError('Invalid checkpoint format.')


def load_model(model_path, model_name, num_classes, device):
    print(f'Creating model: {model_name}')
    if model_name in REGISTRY:
        model = model_for(model_name, num_classes, False)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    try:
        checkpoint = load_checkpoint(model_path)
    except TypeError:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state = state_dict_from_checkpoint(checkpoint)
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        print('\nERROR: Model architecture and checkpoint do not match.')
        print('Set --model_name to the registry alias used during training.')
        print(f'Model name you used: {model_name}')
        raise error
    model.to(device)
    model.eval()
    return model


def export_to_onnx(model, output_path, input_size, batch_size, device):
    print(f'Exporting to ONNX: {output_path}')
    dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    print('ONNX model exported successfully.')


def export_to_tensorrt(onnx_path, output_path, input_size=224, fp16=False, int8=False, max_batch=4):
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            'TensorRT is not installed. Use --onnx-only, or install TensorRT on the target GPU machine.'
        ) from error

    logger = trt.Logger(trt.Logger.WARNING)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, builder.create_network(flag) as network, \
            trt.OnnxParser(network, logger) as parser:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print('Using TensorRT FP16')
        if int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print('Using TensorRT INT8')
        print(f'Parsing ONNX: {onnx_path}')
        with open(onnx_path, 'rb') as handle:
            if not parser.parse(handle.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError('Failed to parse the ONNX model')
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        min_shape = (1, 3, input_size, input_size)
        profile.set_shape(input_name, min=min_shape, opt=min_shape,
                          max=(max_batch, 3, input_size, input_size))
        config.add_optimization_profile(profile)
        print('Building serialized TensorRT engine...')
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('Failed to build TensorRT engine.')
        Path(output_path).write_bytes(serialized)
        print('TensorRT engine exported successfully.')


def onnx_session(onnx_path, device):
    import onnxruntime as ort
    available = ort.get_available_providers()
    if str(device).startswith('cuda') and 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print('ONNX Runtime providers: CUDAExecutionProvider, CPUExecutionProvider')
    else:
        providers = ['CPUExecutionProvider']
        if str(device).startswith('cuda'):
            print('CUDAExecutionProvider unavailable; using CPUExecutionProvider. Install onnxruntime-gpu.')
        else:
            print('ONNX Runtime providers: CPUExecutionProvider')
    return ort.InferenceSession(onnx_path, providers=providers)


def preprocess_image(path, input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(MEAN), std=list(STD)),
    ])
    image = Image.open(path).convert('RGB')
    return transform(image).unsqueeze(0).numpy().astype(np.float32)


def test_onnx_inference(onnx_path, test_image_path, class_names, device, input_size=224):
    print('Testing ONNX inference...')
    session = onnx_session(onnx_path, device)
    input_tensor = preprocess_image(test_image_path, input_size)
    outputs = session.run(None, {'input': input_tensor})
    prediction = int(np.argmax(outputs[0], axis=1)[0])
    print(f'ONNX prediction: {class_names[prediction]}')
    return prediction


def test_tensorrt_inference(engine_path, test_image_path, class_names, input_size=224):
    print('Testing TensorRT inference...')
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    context = engine.create_execution_context()
    input_tensor = preprocess_image(test_image_path, input_size)
    output = np.empty((1, len(class_names)), dtype=np.float32)
    d_input = cuda.mem_alloc(input_tensor.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)
    cuda.memcpy_htod(d_input, input_tensor)
    context.execute_v2(bindings=[int(d_input), int(d_output)])
    cuda.memcpy_dtoh(output, d_output)
    prediction = int(np.argmax(output, axis=1)[0])
    print(f'TensorRT prediction: {class_names[prediction]}')
    return prediction


def main():
    args = parse_args()
    if args.lock_dir:
        inferred = resolve_from_lock(args.lock_dir)
        args.model_path = args.model_path or inferred['model_path']
        args.model_name = inferred['model_name'] or args.model_name
        if inferred['num_classes']:
            args.num_classes = inferred['num_classes']
        if inferred['class_names'] and not args.class_names:
            args.class_names = inferred['class_names']
        if args.output_dir == 'exported_models':
            args.output_dir = inferred['output_dir']
    if not args.model_path:
        raise ValueError('Provide --model_path or --lock-dir.')

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    class_names = args.class_names or [f'class_{i}' for i in range(args.num_classes)]
    if len(class_names) != args.num_classes:
        print(f'Updating num_classes to match class names: {len(class_names)}')
        args.num_classes = len(class_names)
    print(f'Class names: {class_names}')
    print(f'Loading model from: {args.model_path}')
    model = load_model(args.model_path, args.model_name, args.num_classes, device)

    stem = Path(args.model_path).stem
    onnx_path = os.path.join(args.output_dir, f'{stem}.onnx')
    tensorrt_path = os.path.join(args.output_dir, f'{stem}.engine')
    export_to_onnx(model, onnx_path, args.input_size, args.batch_size, device)
    if not args.onnx_only:
        export_to_tensorrt(onnx_path, tensorrt_path, args.input_size, args.fp16, args.int8)

    if os.path.exists(args.test_image):
        print('\nTesting exported model inference...')
        test_onnx_inference(onnx_path, args.test_image, class_names, device, args.input_size)
        if not args.onnx_only and os.path.exists(tensorrt_path):
            test_tensorrt_inference(tensorrt_path, args.test_image, class_names, args.input_size)

    print('\nExport completed.')
    print(f'ONNX model: {onnx_path}')
    if not args.onnx_only:
        print(f'TensorRT engine: {tensorrt_path}')


if __name__ == '__main__':
    main()
