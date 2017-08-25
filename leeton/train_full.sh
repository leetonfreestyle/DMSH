#!/usr/bin/env sh

TOOLS=./build/tools
netname=$1

#nohup $TOOLS/caffe train \
#    --solver=leeton/DSH_NUS48bit_solver.prototxt -gpu 0 > leeton/log.txt 2>&1 & 

sed "s/NETNAME/${netname}/g" leeton/solver_example.prototxt > leeton/solver.prototxt
nohup $TOOLS/caffe train \
    --solver=leeton/solver.prototxt > leeton/log_${netname}.txt 2>&1 &

# nohup $TOOLS/caffe train \
#     --solver=leeton/DSH_NUS12bit_solver.prototxt -gpu 0 \
#     --snapshot=leeton/snapshots/DSH_NUS12bit_iter_50000.solverstate  \
#     > leeton/log.txt 2>&1 & 
