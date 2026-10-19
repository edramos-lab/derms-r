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
    # Export + deploy into a Triton model repository as "cifar10_efficientvit_b0":
    python export_models.py --lock-dir /workspace/results/runs/<job_id>/cifar10/locked \\
        --onnx-only --deploy-triton /workspace/triton-models
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
                        help='locked/ directory; infers checkpoint, alias, class count, and dataset')
    parser.add_argument('--class_names', type=str, nargs='+',
                        help='Class names (space-separated). Inferred from lock.json when --lock-dir is set.')
    parser.add_argument('--test_image', type=str, default='test_image.jpg')
    parser.add_argument('--deploy-triton', type=Path, default=None,
                        help='Triton model-repository root; if set, places the exported ONNX model at '
                             '{root}/{dataset}_{architecture}/1/model.onnx with a generated config.pbtxt '
                             'and labels.txt. Requires --lock-dir (for the dataset name) or --dataset.')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name for --deploy-triton when not using --lock-dir.')
    parser.add_argument('--triton_max_batch', type=int, default=8)
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
    # lock.json stores the dataset name directly; fall back to the directory
    # layout (.../runs/{job_id}/{dataset}/locked) if an older lock.json lacks it.
    dataset = payload.get('dataset') or lock_dir.parent.name
    return {
        'model_path': str(checkpoint),
        'model_name': alias,
        'num_classes': len(classes) if classes else int(winner.get('number_of_folds') or 0) or None,
        'class_names': classes,
        'dataset': dataset,
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
    """Export a PyTorch model to ONNX.

    Always traces on CPU, regardless of `device`, and disables timm's fused
    scaled-dot-product attention first. Both are correctness fixes, not
    performance choices -- confirmed (in a sibling project using the same
    timm architectures: efficientnet_b0, convnextv2_tiny, deit3_base, ...)
    to silently produce wrong predictions otherwise, not just slower ones:

    1. GPU-traced ONNX graphs bake in cuDNN-specific kernel/op-ordering
       decisions from that one forward pass that don't reproduce correctly
       once the graph runs through onnxruntime/TensorRT instead of PyTorch's
       own CUDA kernels. Re-exporting the identical checkpoint on CPU matched
       PyTorch to ~1e-6; the GPU-traced export had a real, large (>1.0) logit
       difference. Random-noise smoke-test inputs look fine either way, which
       is why this is easy to miss.
    2. timm's fused SDPA path exports the attention scale as a dynamic,
       shape-computed subgraph (Shape -> Gather -> Cast -> Sqrt -> Div)
       instead of a baked-in constant. TensorRT's graph optimizer mishandles
       that, washing every attention block's softmax out to near-uniform
       output. Confirmed on deit3_base_patch16_224 (this registry's
       deit3_base): 72% -> 99.7% accuracy after disabling fused attention,
       with FP16 and FP32 equally broken beforehand, ruling out precision as
       the cause. No-op for non-attention (non-timm) architectures.
    """
    from timm.layers import set_fused_attn
    set_fused_attn(False)
    cpu_model = model.eval().to('cpu')
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    print(f'Exporting to ONNX: {output_path}')
    torch.onnx.export(
        cpu_model,
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

    # The dynamo-based exporter path can externalize weights into a
    # same-directory "<name>.onnx.data" file even for small models, whose
    # location is stored in the graph as the literal relative name
    # "model.onnx.data" regardless of what output_path is actually called.
    # When multiple models land in the same flat directory (e.g. this
    # script's --output_dir, or a Triton model repository), the second
    # export silently overwrites the first model's weight data. Every model
    # here is well under the 2GB protobuf single-file limit, so fold any
    # external data back into one self-contained file.
    data_path = output_path + '.data'
    if os.path.exists(data_path):
        onnx_model = onnx.load(output_path)  # loads external data by default
        onnx.save_model(onnx_model, output_path, save_as_external_data=False)
        os.remove(data_path)
        print(f'Folded external data ({data_path}) into single-file {output_path}')

    # .to('cpu') above mutates the module in place (cpu_model is model), so
    # move it back to the caller's original device as a courtesy -- nothing
    # here should leave the passed-in model relocated as a side effect.
    model.to(device)
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


def write_triton_config(config_path, model_name, num_classes, input_size=224,
                        max_batch_size=8, backend='onnxruntime'):
    """Generate a Triton config.pbtxt for an exported ONNX model.

    backend='onnxruntime' (not 'tensorrt') by default: the TensorRT engines
    built by export_to_tensorrt() here are tied to the exact TensorRT version
    inside this container, and Triton's own image can pin a different one --
    a sibling project confirmed real "old deserialization call on a newer
    plan file" failures from that mismatch. ONNX Runtime has no such
    versioning coupling and is what Triton's model-repository `.onnx` files
    are served with here.
    """
    config = f"""name: "{model_name}"
backend: "{backend}"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [3, {input_size}, {input_size}]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [{num_classes}]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

dynamic_batching {{
  preferred_batch_size: [1, 4]
  max_queue_delay_microseconds: 100
}}
"""
    Path(config_path).write_text(config)
    print(f'Triton config written to {config_path}')


def write_labels_file(labels_path, class_names):
    Path(labels_path).write_text('\n'.join(class_names) + '\n')
    print(f'Labels written to {labels_path}')


def deploy_to_triton(onnx_path, repo_root, dataset, architecture, num_classes,
                     class_names, input_size=224, max_batch_size=8):
    """Place an exported ONNX model into a Triton model-repository layout.

    Model name is "{dataset}_{architecture}" so multiple datasets and
    architectures can share one repository without colliding (Triton keys
    models by directory name under the repository root).

    Layout Triton expects:
        {repo_root}/{dataset}_{architecture}/config.pbtxt
        {repo_root}/{dataset}_{architecture}/labels.txt
        {repo_root}/{dataset}_{architecture}/1/model.onnx
    """
    model_name = f'{dataset}_{architecture}'
    model_dir = Path(repo_root) / model_name
    version_dir = model_dir / '1'
    version_dir.mkdir(parents=True, exist_ok=True)

    dest_onnx = version_dir / 'model.onnx'
    dest_onnx.write_bytes(Path(onnx_path).read_bytes())
    write_triton_config(model_dir / 'config.pbtxt', model_name, num_classes, input_size, max_batch_size)
    write_labels_file(model_dir / 'labels.txt', class_names)
    print(f'Deployed to Triton model repository: {model_dir}')
    return model_name


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
        args.dataset = args.dataset or inferred['dataset']
    if not args.model_path:
        raise ValueError('Provide --model_path or --lock-dir.')
    if args.deploy_triton and not args.dataset:
        raise ValueError('--deploy-triton requires --dataset (or --lock-dir, which infers it).')

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

    triton_model_name = None
    if args.deploy_triton:
        print('\nDeploying to Triton model repository...')
        triton_model_name = deploy_to_triton(
            onnx_path, args.deploy_triton, args.dataset, args.model_name,
            args.num_classes, class_names, args.input_size, args.triton_max_batch,
        )

    print('\nExport completed.')
    print(f'ONNX model: {onnx_path}')
    if not args.onnx_only:
        print(f'TensorRT engine: {tensorrt_path}')
    if triton_model_name:
        print(f'Triton model repository entry: {args.deploy_triton}/{triton_model_name}')


if __name__ == '__main__':
    main()
