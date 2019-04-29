import logging
import os
import torch
from collections import defaultdict
import numpy as np
from pycocotools import mask as maskUtils

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker


def do_sunrgbd_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    logger.info("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_sunspot_detection(predictions, dataset)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_sunspot_segmentation(predictions, dataset)

    results = defaultdict(dict)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        iou = np.empty(shape=[0,])
        if iou_type == "bbox":
            for pred in coco_results['bbox']:
                gt = dataset.coco.anns[pred['ann_id']]
                iou = np.append(iou,
                                maskUtils.iou([pred['bbox']], [gt['bbox']], [0]).flatten(),
                                axis=0)
        elif iou_type == "segm":
            for pred in coco_results['segm']:
                gt = dataset.coco.anns[pred['ann_id']]
                iou = np.append(iou,
                                maskUtils.iou([pred['segmentation']], [gt['segmentation']], [0]).flatten(),
                                axis=0)

        results[iou_type]['average_iou'] = np.mean(iou)
        results[iou_type]['median_iou'] = np.median(iou)

        print('{} IOU:\n\tmean: {}\n\tmedian: {}'.format(iou_type, results[iou_type]['average_iou'], results[iou_type]['median_iou']))

    logger.info(results)
    return results, coco_results


def prepare_for_sunspot_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    results = []
    for id, prediction in enumerate(predictions):
        original_id = dataset.split_index[id]
        if len(prediction) == 0:
            continue

        image_id = int(original_id.split('_')[1])
        img_info = dataset.coco.imgs[image_id]
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        results.extend(
            [
                {
                    "ann_id": original_id.split('_', 1)[1],
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return results


def prepare_for_sunspot_segmentation(predictions, dataset):
    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    results = []
    for id, prediction in enumerate(predictions):
        original_id = dataset.split_index[id]
        if len(prediction) == 0:
            continue

        image_id = int(original_id.split("_")[1])
        img_info = dataset.coco.imgs[image_id]
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            maskUtils.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        results.extend(
            [
                {
                    "ann_id": original_id.split('_', 1)[1],
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return results



if __name__=="__main__":
    import argparse

    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.data import make_data_loader

    # Try to debug this independently
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/text_experiment.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--prediction_file",
                        default="output/text_experiments_with_depth/inference/sunspot_test/predictions.pth",
                        metavar="FILE",
                        help="path to prediction file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    data_loader = make_data_loader(cfg, split=False, is_distributed=False)
    dataset_name = cfg.DATASETS.TEST[0]

    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    predictions = torch.load(args.prediction_file)

    do_sunrgbd_evaluation(
            data_loader[0].dataset,
            predictions,
            box_only=False,
            output_folder=output_folder,
            iou_types=("bbox", "segm"),
            expected_results=(),
            expected_results_sigma_tol=4,
    )
