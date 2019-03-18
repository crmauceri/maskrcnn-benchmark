"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.modeling.backbone import build_backbone
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.structures.tensorlist import to_tensor_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.modeling.language_models.LSTM import LanguageModel

import datetime
import logging
import time
import itertools


from maskrcnn_benchmark.utils.metric_logger import MetricLogger

class DepthRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(DepthRCNN, self).__init__()
        self.loss_weights = cfg.LOSS_WEIGHTS

        self.image_backbone = build_backbone(cfg)
        if cfg.MODEL.BACKBONE.DEPTH:
            self.hha_backbone = build_backbone(cfg)
            self.rpn = build_rpn(cfg, self.image_backbone.out_channels * 2)
            self.roi_heads = build_roi_heads(cfg, self.image_backbone.out_channels * 2)
        else:
            self.hha_backbone = None
            self.rpn = build_rpn(cfg, self.image_backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.image_backbone.out_channels)


    def features_forward(self, image_list, HHA_list):
        RGB_features = self.image_backbone(image_list.tensors)

        if self.hha_backbone:
            HHA_features = self.hha_backbone(HHA_list.tensors)

            image_features = []
            for i in range(len(RGB_features)):  # Number of anchor boxes?
                # Dimensions are (batch, feature, height, width)
                image_features.append(torch.cat((RGB_features[i], HHA_features[i]), 1))
        else:
            image_features = RGB_features

        return image_features

    def predictions_forward(self, image_list, features, targets):
        proposals, proposal_losses = self.rpn(image_list, features, targets)

        if self.training:
            proposal_losses['loss_objectness'] *= self.loss_weights.loss_objectness
            proposal_losses['loss_rpn_box_reg'] *= self.loss_weights.loss_rpn_box_reg

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return result, losses

    def instance_prep(self, instance, device, targets):
        images, HHAs = instance

        images = images.to(device)
        image_list = to_image_list(images)


        if self.hha_backbone:
            HHAs = HHAs.to(device)
            HHA_list = to_image_list(HHAs)
        else:
            HHA_list = None

        if targets is not None:
            targets = [target.to(device) for target in targets]

        return image_list, HHA_list, targets


    def forward(self, instance, device, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        image_list, HHA_list = self.instance_prep(instance, device, targets)

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        image_features = self.features_forward(image_list, HHA_list)

        result, losses = self.predictions_forward(image_list, image_features, targets)

        if self.training:
            return losses

        return result


class ReferExpRCNN(DepthRCNN):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(ReferExpRCNN, self).__init__(cfg)
        self.loss_weights = cfg.LOSS_WEIGHTS

        # Text Embedding Network
        self.wordnet = LanguageModel(cfg)

        # Ref Localization Network
        if self.hha_backbone:
            self.ref_rpn = build_rpn(cfg, self.image_backbone.out_channels * 2 + 1024)
            self.ref_roi_heads = build_roi_heads(cfg, self.image_backbone.out_channels * 2 + 1024)
        else:
            self.ref_rpn = build_rpn(cfg, self.image_backbone.out_channels + 1024)
            self.ref_roi_heads = build_roi_heads(cfg, self.image_backbone.out_channels + 1024)

    def instance_prep(self, instance, device, seg_targets):
        images, HHAs, sentences = instance
        images, HHAs, seg_targets = super().instance_prep((images, HHAs), device, seg_targets)

        sentences = [s.to(device) for s in sentences]
        ref_targets = []
        if seg_targets is not None:
            for ind, s in enumerate(sentences):
                s.trim()
                s_target = []
                for ann in s.get_field('ann_id'):
                    s_target.append(seg_targets[ind].get_field('ann_id').index(ann))
                ref_targets.extend(seg_targets[ind][torch.tensor(s_target)].to_list())
        else:
            ref_targets = None

        return images, HHAs, sentences, seg_targets, ref_targets

    def predictions_forward(self, image_list, features, targets):
        proposals, proposal_losses = self.ref_rpn(image_list, features, targets)

        if self.training:
            proposal_losses['loss_objectness'] *= self.loss_weights.ref_objectness
            proposal_losses['loss_rpn_box_reg'] *= self.loss_weights.ref_rpn_box_reg

        if self.ref_roi_heads:
            x, result, detector_losses = self.ref_roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # Change key names
        keys = losses.keys()
        key_dict = dict(zip(keys, ['refexp_'+k for k in keys]))
        losses = {key_dict[key]: value for (key,value) in losses.items()}

        return result, losses

    def forward(self, instance, device, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        image_list, HHA_list, sentences, seg_targets, ref_targets = self.instance_prep(instance, device, targets)

        # Calculate image features
        image_features = self.features_forward(image_list, HHA_list)

        # Calculate text features
        sentence_batch = to_tensor_list(sentences)
        self.wordnet.clear_gradients(batch_size=len(sentence_batch))
        sentence_targets = sentence_batch.get_target()

        text_prediction = self.wordnet(sentence_batch)
        text_loss = self.wordnet.loss_function(text_prediction, sentence_targets)
        text_features = self.wordnet.hidden[0]
        text_shape = text_features.shape

        full_feature = [0]*len(image_features)
        image_sizes = []
        for i, feature in enumerate(image_features):
            image_shape = feature.shape
            full_shape = torch.Size((text_shape[1], image_shape[1]+text_shape[2], image_shape[2], image_shape[3]))
            f = torch.zeros(full_shape, device=device)

            f[:, image_shape[1]:, :, :] = text_features.reshape((text_shape[1], text_shape[2], 1, 1)).repeat(1, 1, image_shape[2], image_shape[3])
            image_mask = [len(s) for s in sentences]
            for j, repeats in enumerate(image_mask):
                f[sum(image_mask[:j]):sum(image_mask[:j+1]), :image_shape[1], :, :] = feature[j, :, :, :].repeat(repeats, 1, 1, 1)
                image_sizes.extend(list(itertools.repeat(image_list.image_sizes[j], repeats)))
            full_feature[i] = f

        ## Losses and predictions
        result, ref_exp_loss = self.predictions_forward(image_sizes, full_feature, ref_targets)

        if self.training:
            # Normal instance segmentation
            losses = super().predictions_forward(image_list, image_features, seg_targets)[1]

            # Language model
            losses['text_loss'] = text_loss

            # Referring Expression Localization
            losses.update(ref_exp_loss)
            return losses

        return result