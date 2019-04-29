#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/refcocog_refexprcnn_classification_model.yaml SOLVER.IMS_PER_BATCH 4 #MODEL.DEVICE cpu
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/refcocog_refexprcnn_language_model.yaml SOLVER.IMS_PER_BATCH 4 #MODEL.DEVICE cpu