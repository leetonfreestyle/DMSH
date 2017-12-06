#!/usr/bin/env sh
for i in 10000 20000 30000;do
    rm leeton/code.dat
    rm leeton/label.dat
    
    build/tools/extract_features_binary leeton/snapshots/finetune_iter_$i.caffemodel leeton/finetune.prototxt ip1_f leeton/code.dat 10 0
    build/tools/extract_features_binary leeton/snapshots/finetune_iter_$i.caffemodel leeton/finetune.prototxt label leeton/label.dat 10 0
    python leeton/eval_WmAP.py leeton/code.dat leeton/label.dat -batchsize 100 -nbatch 100 -nlabel 21 -nbit 48 
done
