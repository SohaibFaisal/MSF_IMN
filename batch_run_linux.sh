#!/bin/bash

GPU_ID=0
EPOCHS=150
LAYERS=5
NODES=2
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --layers $LAYERS \
            --nodes $NODES \
            --epochs $EPOCHS

