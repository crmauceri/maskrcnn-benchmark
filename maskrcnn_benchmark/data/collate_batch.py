# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


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

class RefExpBatchCollator(object):
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
        HHAs = to_image_list(transposed_batch[1], self.size_divisible)
        targets = transposed_batch[3]
        img_ids = transposed_batch[4]

        #TODO these need work. There are multiple sentences per image and currently they are not properly associated with
        # bbox or segments and they are getting scrambeled by collation
        sentences = transposed_batch[2]
        sent_ids = transposed_batch[5]
        return images, HHAs, sentences, targets, img_ids, sent_ids
