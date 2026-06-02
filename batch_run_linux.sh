#!/bin/bash

GPU_ID=2
EPOCHS=100

for LAYERS in 3 4 5
do
    for NODES in 1 2 3
    do
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --layers $LAYERS \
            --nodes $NODES \
            --epochs $EPOCHS
    done
done
