#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/LSTM_classification_experiment.yaml SOLVER.IMS_PER_BATCH 4 #MODEL.DEVICE cpu
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/LSTM_language_experiment.yaml SOLVER.IMS_PER_BATCH 4 #MODEL.DEVICE cpu
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/refcoco_segmentation.yaml SOLVER.IMS_PER_BATCH 4 #MODEL.DEVICE cpu