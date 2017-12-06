#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=/home/leeton/dataSet/NUS
DATA=leeton
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/leeton/dataSet/NUS/
VAL_DATA_ROOT=/home/leeton/dataSet/NUS/

RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=64
  RESIZE_WIDTH=64
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/NUS_trainList.txt \
    $EXAMPLE/train_lmdb \
    21

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $VAL_DATA_ROOT \
    $DATA/NUS_testList.txt \
    $EXAMPLE/test_lmdb \
    21

echo "Done."
