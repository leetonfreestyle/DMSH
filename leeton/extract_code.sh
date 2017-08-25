#!/usr/bin/env sh

# for i in 130000 140000 150000;do
i=$1
rm leeton/code.dat
rm leeton/label.dat

build/tools/extract_features_binary leeton/snapshots/DSH_NUS12bit_iter_$i.caffemodel leeton/DSH_NUS12bit.prototxt ip1 leeton/code.dat 10 0
build/tools/extract_features_binary leeton/snapshots/DSH_NUS12bit_iter_$i.caffemodel leeton/DSH_NUS12bit.prototxt label leeton/label.dat 10 0
python leeton/eval_WmAP.py leeton/code.dat leeton/label.dat -batchsize 1000 -nbatch 10 -nlabel 21 -nbit 24 
# done
