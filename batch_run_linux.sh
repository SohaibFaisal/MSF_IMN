#!/bin/bash

GPU_ID=0
EPOCHS=500
LAYERS=5
NODES=2
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --layers $LAYERS \
            --nodes $NODES \
            --epochs $EPOCHS

