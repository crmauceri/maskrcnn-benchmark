#!/usr/bin/env bash

# Language Models
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/LSTM_classification_experiment.yaml SOLVER.IMS_PER_BATCH 4 MODEL.DEVICE cpu
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/LSTM_language_experiment.yaml SOLVER.IMS_PER_BATCH 4 MODEL.DEVICE cpu

# Segmentation Models
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/refcoco_segmentation.yaml SOLVER.IMS_PER_BATCH 4 MODEL.DEVICE cpu
# CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/refcoco_depth_segmentation.yaml SOLVER.IMS_PER_BATCH 4 MODEL.DEVICE cpu

# Referring Expression Models
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/refcocog_refexprcnn_classification_model.yaml SOLVER.IMS_PER_BATCH 4 #MODEL.DEVICE cpu
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/refcocog_refexprcnn_language_model.yaml SOLVER.IMS_PER_BATCH 4 #MODEL.DEVICE cpu