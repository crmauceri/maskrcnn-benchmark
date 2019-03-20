# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.tensorlist import to_tensor_list
from torch import Tensor


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids

class HHACollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        batch = [(b[0][0], b[0][1], b[1], b[2]) for b in batch]
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        if isinstance(transposed_batch[1][0], Tensor):
            HHAs = to_image_list(transposed_batch[1], self.size_divisible)
        else:
            HHAs = transposed_batch[1]
        targets = transposed_batch[2]
        img_ids = transposed_batch[3]

        return (images, HHAs), targets, img_ids

class RefExpBatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        batch = [(b[0][0], b[0][1], b[0][2], b[1], b[2]) for b in batch]
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        if isinstance(transposed_batch[1][0], Tensor):
            HHAs = to_image_list(transposed_batch[1], self.size_divisible)
        else:
            HHAs = transposed_batch[1]
        targets = transposed_batch[3]
        img_ids = transposed_batch[4]
        sentences = transposed_batch[2]
        return (images, HHAs, sentences), targets, img_ids
