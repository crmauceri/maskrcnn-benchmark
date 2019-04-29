# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .generalized_rcnn import GeneralizedRCNN
from .refexp_localizer import ReferExpRCNN, ReferExpRCNN_Old, DepthRCNN
from .LSTM import ClassificationModel, LanguageModel

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "RefExpRCNN": ReferExpRCNN, "DepthRCNN": DepthRCNN,
                                 "TextClassification": ClassificationModel, "LanguageModel": LanguageModel,
                                 "RefExpRCNN_Old": ReferExpRCNN_Old}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
