import unittest, torch

from maskrcnn_benchmark.data.datasets.evaluation.sunspot.sunspot_eval import do_sunrgbd_evaluation
from maskrcnn_benchmark.data.datasets.sunspot import ReferExpressionDataset


class TestEval(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load ground truth
        ann_file = '../../datasets/sunspot/annotations/instances.json'
        img_root = '../../datasets/sunspot/images'
        ref_file = '../../datasets/sunspot/annotations/refs(boulder).p'
        vocab_file = '../../datasets/vocab_file.txt'
        self.refer = ReferExpressionDataset(ann_file, img_root, ref_file, vocab_file, True, active_split="test")

        self.gt = []
        for instance in self.refer:
            (img, hha, sents), target, idx = instance

            original_id = target.get_field('ann_id')[0]
            image_id = int(original_id.split("_")[0])
            img_info = self.refer.coco.imgs[image_id]
            image_width = img_info["width"]
            image_height = img_info["height"]

            bbox_list = target[
                torch.tensor([t in sents.get_field('ann_id') for t in target.get_field('ann_id')], dtype=torch.uint8)]
            bbox_list.add_field('scores', torch.ones(bbox_list.bbox.shape[0]))
            segmask = torch.cat([m.mask.unsqueeze(0).unsqueeze(0) for m in bbox_list.get_field('masks').masks], dim=0)

            assert(segmask.shape[-2:] == torch.Size([image_height, image_width]))
            bbox_list.add_field('mask', torch.as_tensor(segmask, dtype=torch.uint8))
            self.gt.append(bbox_list)

    def test_all_correct(self):
        results, cocoresults = do_sunrgbd_evaluation(
            self.refer,
            self.gt,
            box_only=False,
            output_folder='output/test/',
            iou_types=("bbox", "segm"),
            expected_results=(),
            expected_results_sigma_tol=4,
        )

        for label, result in results['bbox'].items():
            self.assertEquals(result, 1.0, "Should be 1.0")

        for label, result in results['segm'].items():
            self.assertEquals(result, 1.0, "Should be 1.0")


if __name__ == '__main__':
    unittest.main()