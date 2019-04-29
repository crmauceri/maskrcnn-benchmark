import torch

import os.path as osp
import os
import pickle
from PIL import Image
from random import randint

from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.structures.tensorlist import TensorList
from torchvision.transforms import functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list


class HHADataset(COCODataset):
    def __init__(
        self, ann_file, img_root, remove_images_without_annotations, transforms=None, has_depth=True, depth_root=None,
    ):
        super().__init__(ann_file, img_root, remove_images_without_annotations, transforms)

        # Set class variables
        self.has_depth = has_depth
        self.depth_root = depth_root


    def __getitem__(self, idx):
        return self.getItem(idx)

    def getItem(self, idx):
        img, target, image_idx = super().__getitem__(idx)
        if self.has_depth:
            hha = self.loadHHA(image_idx)
        else:
            hha = None

        return (img, hha), target, image_idx

    def loadHHA(self, img_id):
        img_data = self.coco.imgs[img_id]

        if self.depth_root:
            path = osp.join(self.depth_root, self.coco.loadImgs(img_id)[0]['file_name']).replace('jpg', 'png')
        else:
            dir = self.coco.loadImgs(img_id)[0]['file_name'].split('image')[0]
            file = [file for file in os.listdir(osp.join(self.root, dir, 'HHA')) if file.endswith('png')][-1]
            path = osp.join(self.root, dir, 'HHA', file)

        img = Image.open(path).convert('RGB')
        img = F.resize(img, (img_data['height'], img_data['width']))
        if self.transforms is not None:
            img = self.transforms(img, None)[0]

        return img


class ReferExpressionDataset(HHADataset):
    def __init__(
        self, ann_file=None, img_root=None, ref_file=None, vocab_file=None, remove_images_without_annotations=False, \
            transforms=None, active_split=None, has_depth=False, exclude_list=[], depth_root=None,
    ):
        super().__init__(ann_file, img_root, remove_images_without_annotations, transforms, has_depth, depth_root)

        # Set class variables
        self.active_split = active_split
        self.exclude_list = exclude_list

        if vocab_file is not None:
            self.load_vocabulary(vocab_file)
        else:
            self.vocab =[]
            self.word2idx = []

        # Index referring expressions
        if ref_file is not None:
            self.createRefIndex(ref_file)
        else:
            self.split_index = []

    def load_vocabulary(self, vocab_file):
        # Initialize vocabulary
        with open(vocab_file, 'r') as f:
            self.vocab = [v.strip() for v in f.readlines()]
        self.vocab.extend(['<bos>', '<eos>', '<unk>'])
        self.word2idx = dict(zip(self.vocab, range(1, len(self.vocab) + 1)))

    def __len__(self):
        if self.active_split == 'train':
            return len(self.train_index)
        elif self.active_split == 'test':
            return len(self.test_index)
        elif self.active_split == 'val':
            return len(self.val_index)
        else:
            raise ValueError("No active split")

    def __getitem__(self, idx):
        if self.active_split == 'train':
            self.split_index = self.train_index
        elif self.active_split == 'test':
            self.split_index = self.test_index
        elif self.active_split == 'val':
            self.split_index = self.val_index
        else:
            raise ValueError("No active split")

        sent_idx = self.split_index[idx]
        img_idx = int(self.coco.sentToRef[sent_idx]['image_id'])
        img_objs, target, img_idx = super().getItem(img_idx)
        img, hha = img_objs

        sentence = self.coco.sents[sent_idx]

        sents = TensorList([sentence['vocab']])
        sents.add_field('tokens', [sentence['tokens']])
        sents.add_field('img_id', [img_idx])

        ann_id = self.coco.sentToRef[sent_idx]['ann_id']
        sents.add_field('ann_id', [ann_id])

        ann_target = [t for t in target if ann_id in t.get_field('ann_id')]
        # Might be an issue with sentences without corresponding ann targets
        assert len(ann_target) > 0
        sents.add_field('ann_target', ann_target)

        bbox_list = target[
            torch.tensor([t in sents.get_field('ann_id') for t in target.get_field('ann_id')], dtype=torch.uint8)]
        bbox_list.add_field('scores', torch.ones(bbox_list.bbox.shape))

        return (img, hha, sents), target, self.split_index[idx]

    def createRefIndex(self, ref_file):

        with open(ref_file, 'rb') as f:
            refs = pickle.load(f)

        print('creating index...')

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}

        refs = [ref for ref in refs if ref['ann_id'] in self.coco.anns]
        for ref in refs:
            # ids
            ref_id = ref['ref_id']

            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping of sent
            for sent in ref['sentences']:
                if sent['sent_id'] not in self.exclude_list:
                    self.sent2vocab(sent)
                    Sents[sent['sent_id']] = sent
                    sentToRef[sent['sent_id']] = ref
                    sentToTokens[sent['sent_id']] = sent['tokens']
                    sent['split'] = ref['split']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = self.coco.anns[ann_id]
            annToRef[ann_id] = ref

        # create class members
        self.coco.refs = Refs
        self.coco.imgToRefs = imgToRefs
        self.coco.refToAnn = refToAnn
        self.coco.annToRef = annToRef
        self.coco.catToRef = catToRefs
        self.coco.sents = Sents
        self.coco.sentToRef = sentToRef
        self.coco.sentToTokens = sentToTokens

        self.max_sent_len = max(
            [len(sent['tokens']) for sent in self.coco.sents.values()]) + 2  # For the begining and end tokens

        #This is the coco object, image id index
        self.ids = dict(zip(self.coco.imgs.keys(), self.coco.imgs.keys()))

        self.train_index = [sent_id for sent_id, sent in self.coco.sents.items() if sent['split'] == 'train']
        self.train_index.sort()

        self.val_index = [sent_id for sent_id, sent in self.coco.sents.items() if sent['split'] == 'val']
        self.val_index.sort()

        self.test_index = [sent_id for sent_id, sent in self.coco.sents.items() if sent['split'] == 'test']
        self.test_index.sort()

    def sent2vocab(self, sent):
        begin_index = self.word2idx['<bos>']
        end_index = self.word2idx['<eos>']
        unk_index = self.word2idx['<unk>']

        sent['vocab'] = [begin_index]
        for token in sent['tokens']:
            if token in self.word2idx:
                sent['vocab'].append(self.word2idx[token])
            else:
                sent['vocab'].append(unk_index)
        sent['vocab'].append(end_index)

    def get_img_info(self, index):

        if self.active_split == 'train':
            self.split_index = self.train_index
        elif self.active_split == 'test':
            self.split_index = self.test_index
        elif self.active_split == 'val':
            self.split_index = self.val_index

        try:
            img_id = int(self.coco.sentToRef[self.split_index[index]]['image_id'])
            img_data = self.coco.imgs[img_id]
        except KeyError as e:
            raise(e)
        return img_data

    @staticmethod
    def instance_prep(self, instance, device, targets, use_HHA=True, training=True):
        images, HHAs, sentences = instance

        images = images.to(device)
        image_list = to_image_list(images)

        if use_HHA:
            HHAs = HHAs.to(device)
            HHA_list = to_image_list(HHAs)
        else:
            HHA_list = None

        if targets is not None:
            targets = [target.to(device) for target in targets]

        ref_targets = []
        if training:
            if targets is not None:
                for ind, s in enumerate(sentences):
                    s.trim()
                    ref_targets.extend(s.get_field('ann_target'))
            else:
                ref_targets = None

            ref_targets = [t.to(device) for t in ref_targets]

        return image_list, HHA_list, sentences, targets, ref_targets

if __name__ == "__main__":

    from tqdm import tqdm

    # Run through whole dataset once
    ann_file = '../../datasets/sunspot/annotations/instances.json'
    img_root = '../../datasets/sunspot/images'
    ref_file = '../../datasets/sunspot/annotations/refs(boulder).p'
    vocab_file = '../../datasets/vocab_file.txt'
    refer = ReferExpressionDataset(ann_file, img_root, ref_file, vocab_file, True, active_split="val")

    for i in tqdm(range(len(refer))):
        try:
            img, hha, sents, target, idx = refer[i]
        except ValueError as e:
            print(e)
            print(refer.split_index[i])
        except AssertionError as e:
            print(refer.split_index[i])
