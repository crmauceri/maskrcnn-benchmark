import argparse
import torch
import numpy as np

from maskrcnn_benchmark.config import get_cfg_defaults
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from pycocotools import mask as maskUtils

from tqdm import tqdm


def IOU(dt, gt):
    # Segmentation
    if isinstance(dt, torch.Tensor) and isinstance(gt, torch.Tensor):
        assert (dt.shape == gt.shape)
        intersection = torch.sum((dt > 0) & (gt > 0))
        union = torch.sum((dt > 0) | (gt > 0))

    # Bounding Boxes
    elif isinstance(dt, list) and isinstance(gt, list):
        assert (len(dt) == 4 and len(gt) == 4)

        #   |------|  dt
        # |------|     gt
        if gt[0] <= dt[0] <= gt[0] + gt[2] <= dt[0] + dt[2]:
            overlap_x = (gt[0] + gt[2]) - dt[0]

        #   |---|     dt
        # |-------|   gt
        elif gt[0] <= dt[0] <= dt[0] + dt[2] <= gt[0] + gt[2]:
            overlap_x = dt[2]

        # |------|     dt
        #     |------| gt
        elif dt[0] <= gt[0] <= dt[0] + dt[2] <= gt[0] + gt[2]:
            overlap_x = (dt[0] + dt[2]) - gt[0]

        # |-----| |-----|
        else:
            overlap_x = 0

        #   |------|  dt
        # |------|     gt
        if gt[1] <= dt[1] <= gt[1] + gt[3] <= dt[1] + dt[3]:
            overlap_y = (gt[1] + gt[3]) - dt[1]

        #   |---|     dt
        # |-------|   gt
        elif gt[1] <= dt[1] <= dt[1] + dt[3] <= gt[1] + gt[3]:
            overlap_y = dt[3]

        # |------|     dt
        #     |------| gt
        elif dt[1] <= gt[1] <= dt[1] + dt[3] <= gt[1] + gt[3]:
            overlap_y = (dt[1] + dt[3]) - gt[1]

        # |-----| |-----|
        else:
            overlap_y = 0

        intersection = overlap_x * overlap_y
        union = dt[2] * dt[3] + gt[2] * gt[3] - intersection

    else:
        raise ValueError('IOU not implemented for this data type')

    return float(intersection) / float(union)


# Most of this code is from demos/predictor.py
class SegmentationHelper(object):

    def __init__(
            self,
            cfg,
            confidence_threshold=0.7,
            show_mask_heatmaps=False,
            masks_per_dim=2,
            min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def run_on_image(self, image_tensor):
        """
        Arguments:
            image_tensor (ImageList): an image as loaded by DataLoader

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """

        predictions = self.compute_prediction(image_tensor)
        top_predictions = self.select_top_predictions(predictions)

        return top_predictions

    def compute_prediction(self, image_tensor):
        """
        Arguments:
            image_tensor (ImageList): an ImageList as loaded by dataloader

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_tensor, self.device)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # reshape prediction (a BoxList) into the original image size
        for index in range(len(predictions)):
            prediction = predictions[index]
            height, width = image_tensor.image_sizes[index]
            prediction = prediction.resize((width, height))

            if prediction.has_field("mask"):
                # if we have masks, paste the masks in the right position
                # in the image, as defined by the bounding boxes
                masks = prediction.get_field("mask")
                # always single image is passed at a time
                masks = self.masker([masks], [prediction])[0]
                prediction.add_field("mask", masks)

            predictions[index] = prediction

        return predictions

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        for index, prediction in enumerate(predictions):
            scores = prediction.get_field("scores")
            keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
            prediction = prediction[keep]
            scores = prediction.get_field("scores")
            _, idx = scores.sort(0, descending=True)
            predictions[index] = prediction[idx]

        return predictions


def main(cfg_text, cfg_segment):
    # Load saved LSTM network
    language_model = build_detection_model(cfg_text)
    language_model.to(cfg_text.MODEL.DEVICE)

    output_dir = cfg_text.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg_text, language_model, save_dir=output_dir)
    _ = checkpointer.load(cfg_text.MODEL.WEIGHT)
    language_model.eval()

    # Load saved segmentation network
    seg_model = SegmentationHelper(cfg_segment)

    fine_gt = []

    seg_iou = []
    bbox_iou = []

    data_loaders = make_data_loader(cfg_text, split=False, is_distributed=False)
    for index, instance in tqdm(enumerate(data_loaders[0])):
        with torch.no_grad():
            prediction = language_model(instance[0], device=cfg_text.MODEL.DEVICE)
            segmentation_prediction = seg_model.run_on_image(instance[0][0])

        _, pred_ind = prediction[:, -1, :].max(1)
        for j in range(len(pred_ind)):
            segs = segmentation_prediction[j]
            label = pred_ind[j]

            label_mask = segs.get_field('labels') == label
            if any(label_mask):
                score, top_ind = segs[label_mask].get_field('scores').max(0)
                top_seg = segs[label_mask][top_ind]

                ann_seg = instance[0][2][j].get_field('ann_target')[0]

                fine_gt.append(ann_seg.get_field('labels').item())

                bbox_iou.append(IOU(top_seg.bbox.tolist()[0], ann_seg.bbox.tolist()[0]))
                if top_seg.has_field('mask'):
                    top_mask = top_seg.get_field('mask').squeeze()
                    ann_mask = ann_seg.get_field('masks').masks[0].mask
                    seg_iou.append(IOU(top_mask, ann_mask))
                else:
                    seg_iou.append(0.0)
            else:
                bbox_iou.append(0.0)
                seg_iou.append(0.0)

    print("Mean Segmentation IOU: {}".format(np.mean(seg_iou)))
    print("Mean Bounding Box IOU: {}".format(np.mean(bbox_iou)))

    print("\n Class \t Seg IOU \t BBox IOU \t Support")
    for label in data_loaders[0].dataset.coco.cats.values():
        mask = torch.Tensor(fine_gt) == label['id']
        seg_iou = torch.Tensor(seg_iou)
        bbox_iou = torch.Tensor(bbox_iou)
        print("{} \t {:.2f} \t {:.2f} \t{:d}".format(label['name'], torch.mean(seg_iou[mask]), torch.mean(bbox_iou[mask]), torch.sum(mask)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/LSTM_classification_experiment.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg_text = get_cfg_defaults()
    cfg_text.merge_from_file(args.config_file)
    cfg_text.merge_from_list(args.opts)
    cfg_text.freeze()

    cfg_segment = get_cfg_defaults()
    cfg_segment.merge_from_file("configs/original/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
    cfg_segment.merge_from_list(args.opts)
    cfg_segment.freeze()

    main(cfg_text, cfg_segment)
