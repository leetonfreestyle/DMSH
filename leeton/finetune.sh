#!/usr/bin/env sh

TOOLS=./build/tools

nohup $TOOLS/caffe train \
    --solver=leeton/finetune_solver.prototxt \
    --weights=leeton/zoo/LEETON24bit_0.0 > leeton/log_finetune.txt 2>&1 &
# nohup $TOOLS/caffe train \
#     --solver=leeton/finetune_solver.prototxt \
#     --weights=leeton/zoo/leeton_decay0.62.caffemodel > leeton/log_finetune.txt 2>&1 &
# nohup $TOOLS/caffe train \
#     --solver=leeton/finetune_solver.prototxt \
#     --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel > leeton/log_finetune_caffenet.txt 2>&1 &
