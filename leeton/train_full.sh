#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=leeton/DSH_48bit_solver.prototxt -gpu 0 \
    > leeton/log_DSH_48bit.txt 2>&1 &

# nohup $TOOLS/caffe train \
#     --solver=leeton/solver.prototxt -gpu 0\
#     >> leeton/log_DSH_48bit.txt 2>&1 &

# nohup $TOOLS/caffe train \
#     --solver=leeton/CaffeNet_48bit_solver.prototxt -gpu 0 \
#     >> leeton/log_CaffeNet_fix_48bit.txt 2>&1 &

#$TOOLS/caffe train \
#    --solver=CIFAR-10/cifar10_full_solver_lr1.prototxt \
#    --snapshot=CIFAR-10/cifar10_full_iter_40000.solverstate

#$TOOLS/caffe train \
#    --solver=CIFAR-10/cifar10_full_solver_lr2.prototxt \
#    --snapshot=CIFAR-10/cifar10_full_iter_160000.solverstate
