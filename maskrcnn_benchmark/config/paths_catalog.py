# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "sunspot": {
            "img_dir": "sunspot/images",
            "ann_file": "sunspot/annotations/instances.json",
            "refer_file": "sunspot/annotations/refs(boulder).p",
            "vocab_file": "vocab_file.txt",
            "has_depth": True,
            "exclude_list": ['32777128_7408_6']  # A few bad apples that have bad annotation mappings
        },
        "sunspotnodepth": {
            "img_dir": "sunspot/images",
            "ann_file": "sunspot/annotations/instances.json",
            "refer_file": "sunspot/annotations/refs(boulder).p",
            "vocab_file": "vocab_file.txt",
            "has_depth": False
        },
        "refcocoggoogle":{
            "img_dir": "coco/images/train2014",
            "ann_file": "coco/refcocog/instances.json",
            "refer_file": "coco/refcocog/refs(google).p",
            "vocab_file": "vocab_file.txt",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        },
        "refcocogumc": {
            "img_dir": "coco/images/train2014",
            "ann_file": "coco/refcocog/instances.json",
            "refer_file": "coco/refcocog/refs(umd).p",
            "vocab_file": "vocab_file.txt",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        },
        "refcoco+": {
            "img_dir": "coco/images/train2014",
            "ann_file": "coco/refcoco+/instances.json",
            "refer_file": "coco/refcoco+/refs(unc).p",
            "vocab_file": "True.txt",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        },
        "refcocogoogle": {
            "img_dir": "coco/images/train2014",
            "ann_file": "coco/refcoco/instances.json",
            "refer_file": "coco/refcoco/refs(google).p",
            "vocab_file": "vocab_file.txt",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        },
        "refcocounc": {
            "img_dir": "coco/images/train2014",
            "ann_file": "coco/refcoco/instances.json",
            "refer_file": "coco/refcoco/refs(unc).p",
            "vocab_file": "vocab_file.txt",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        },
        "sunrgbd_train": {
            "img_dir": "SUNRGBD/images",
            "ann_file": "sunspot/annotations/instances.json"
        },
        "sunrgbd_val": {
            "img_dir": "SUNRGBD/images",
            "ann_file": "sunspot/annotations/instances.json"
        },
        "sunrgbd_test": {
            "img_dir": "SUNRGBD/images",
            "ann_file": "sunspot/annotations/instances.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/images/train2014",
            "ann_file": "coco/annotations/instances_train2014_minus_refcocog.json",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        },
        "coco_2014_val": {
            "img_dir": "coco/images/val2014",
            "ann_file": "coco/annotations/instances_val2014.json",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        },
        "coco_2014_minival": {
            "img_dir": "coco/images/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/images/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json",
            "has_depth": True,
            "depth_root": "coco/images/megadepth/"
        }
    }

    @staticmethod
    def get(name, dataclass):
        if dataclass == 'HHADataset':
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                img_root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                has_depth=attrs["has_depth"],
                depth_root=os.path.join(data_dir, attrs["depth_root"]) if "depth_root" in attrs else None,
            )
            return dict(
                factory="HHADataset",
                args=args,
            )
        elif dataclass == 'ReferExpressionDataset':
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name.split('_')[0]]
            args = dict(
                img_root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
                ref_file=os.path.join(data_dir, attrs["refer_file"]),
                vocab_file=os.path.join(data_dir, attrs["vocab_file"]),
                has_depth=attrs["has_depth"],
                depth_root=os.path.join(data_dir, attrs["depth_root"]) if "depth_root" in attrs else None,
                active_split=name.split('_')[1],
                exclude_list=attrs["exclude_list"] if "exclude_list" in attrs else []
            )
            return dict(
                factory="ReferExpressionDataset",
                args=args,
            )
        elif dataclass == 'COCODataset':
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        else:
            raise RuntimeError("Dataset not available: {}".format(dataclass))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
