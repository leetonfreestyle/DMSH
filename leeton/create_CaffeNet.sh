#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=/home/leeton/dataSet/man_100k/
DATA=leeton
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/leeton/dataSet/man_100k/Train/
VAL_DATA_ROOT=/home/leeton/dataSet/man_100k/Test/

RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=227
  RESIZE_WIDTH=227
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
    $DATA/trainList.txt \
    $EXAMPLE/train_lmdb \
    30

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $VAL_DATA_ROOT \
    $DATA/testList.txt \
    $EXAMPLE/test_lmdb \
    30

echo "Done."
