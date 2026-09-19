#!/usr/bin/env bash
# ==============================================================================
# Build TensorRT Engine for DIAMOND UNet on NVIDIA Jetson
# ==============================================================================

set -e

ONNX_FILE="${1:-diamond_unet_highway.onnx}"
ENGINE_FILE="${2:-diamond_unet_fp16.engine}"
PRECISION="${3:-fp16}"  # fp16 or int8

echo "=========================================================="
echo " Building TensorRT Engine for DIAMOND on Jetson"
echo " ONNX Source: $ONNX_FILE"
echo " Engine Destination: $ENGINE_FILE"
echo " Precision: $PRECISION"
echo "=========================================================="

if ! command -v trtexec &> /dev/null; then
    echo "trtexec not found in PATH. Checking /usr/src/tensorrt/bin/trtexec..."
    if [ -f /usr/src/tensorrt/bin/trtexec ]; then
        TRTEXEC=/usr/src/tensorrt/bin/trtexec
    else
        echo "Error: trtexec not found. Make sure TensorRT is installed."
        exit 1
    fi
else
    TRTEXEC=trtexec
fi

# Highway dimensions: Obs: 12x48x320, Noisy: 3x48x320, Act: 4
# Min Batch: 1, Opt Batch: 16 (or 32), Max Batch: 64

FLAGS="--onnx=$ONNX_FILE --saveEngine=$ENGINE_FILE"
FLAGS="$FLAGS --minShapes=noisy_next_obs:1x3x48x320,c_noise:1,obs:1x12x48x320,act:1x4"
FLAGS="$FLAGS --optShapes=noisy_next_obs:16x3x48x320,c_noise:16,obs:16x12x48x320,act:16x4"
FLAGS="$FLAGS --maxShapes=noisy_next_obs:64x3x48x320,c_noise:64,obs:64x12x48x320,act:64x4"
FLAGS="$FLAGS --builderOptimizationLevel=5"

if [ "$PRECISION" == "fp16" ]; then
    FLAGS="$FLAGS --fp16"
elif [ "$PRECISION" == "int8" ]; then
    FLAGS="$FLAGS --int8 --fp16"
fi

echo "Running trtexec command..."
$TRTEXEC $FLAGS

echo "=========================================================="
echo " TensorRT Engine successfully built: $ENGINE_FILE"
echo "=========================================================="

