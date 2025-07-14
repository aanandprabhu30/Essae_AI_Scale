#!/usr/bin/env python3
"""
RKNN Conversion Script for AIScale MobileNetV3
Run this on a system with RKNN Toolkit installed
"""

import numpy as np
from rknn.api import RKNN

# Configuration
ONNX_MODEL = 'mobilenetv3_produce_simplified.onnx'
RKNN_MODEL = 'mobilenetv3_produce.rknn'
DATASET_PATH = 'calibration_dataset.txt'  # List of image paths for quantization

# Create RKNN object
rknn = RKNN(verbose=True)

# Pre-process config
print('--> Config model')
rknn.config(
    mean_values=[[123.675, 116.28, 103.53]],  # Pre-calculated from ImageNet
    std_values=[[58.395, 57.12, 57.375]],
    quantized_dtype='asymmetric_affine-u8',  # INT8 quantization
    optimization_level=3,
    target_platform='rk3568',
    output_optimize=1
)
print('done')

# Load ONNX model
print('--> Loading model')
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(
    do_quantization=True,
    dataset=DATASET_PATH,  # Calibration dataset
    pre_compile=False
)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export RKNN model
print('--> Export RKNN model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export RKNN model failed!')
    exit(ret)
print('done')

# Optional: Test inference
print('--> Init runtime environment')
ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

# Release
rknn.release()
print(f'Successfully converted to {RKNN_MODEL}')
