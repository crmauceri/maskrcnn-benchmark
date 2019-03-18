import torch

import os.path as osp
import os
import pickle
from PIL import Image
from random import randint

from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.structures.tensorlist import TensorList


class HHADataset(COCODataset):
    def __init__(
        self, ann_file, img_root, remove_images_without_annotations, transforms=None, has_depth=True,
    ):
        super().__init__(ann_file, img_root, remove_images_without_annotations, transforms)

        # Set class variables
        self.has_depth = has_depth


    def __getitem__(self, idx):
        return self.getItem(idx)

    def getItem(self, idx):
        img, target, image_idx = super().__getitem__(idx)
        if self.has_depth:
            hha = self.loadHHA(image_idx)
        else:
            hha = None

        return img, hha, target, image_idx

    def loadHHA(self, img_id):
        dir = self.coco.loadImgs(img_id)[0]['file_name'].split('image')[0]
        file = [file for file in os.listdir(osp.join(self.root, dir, 'HHA')) if file.endswith('png')][-1]
        path = osp.join(self.root, dir, 'HHA', file)

        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img, None)[0]

        return img


class ReferExpressionDataset(HHADataset):
    def __init__(
        self, ann_file, img_root, ref_file, vocab_file, remove_images_without_annotations, \
            transforms=None, active_split=None, has_depth=False,
    ):
        super().__init__(ann_file, img_root, remove_images_without_annotations, transforms, has_depth)

        # Set class variables
        self.active_split = active_split
        self.exclude_list = ['32777128_7408_6']  # A few bad apples that have bad annotation mappings

        # Initialize vocabulary
        with open(vocab_file, 'r') as f:
            self.vocab = [v.strip() for v in f.readlines()]
        self.vocab.extend(['<bos>', '<eos>', '<unk>'])
        self.word2idx = dict(zip(self.vocab, range(1, len(self.vocab) + 1)))

        # Index referring expressions
        self.createRefIndex(ref_file)


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

        img_idx = int(self.split_index[idx].split('_')[1])
        img, hha, target, img_idx = super().getItem(img_idx)

        sentence = self.coco.sents[self.split_index[idx]]

        sents = TensorList([sentence['vocab']])
        sents.add_field('tokens', [sentence['tokens']])
        sents.add_field('img_id', [sentence['sent_id'].split('_')[1]])

        ann_id = sentence['sent_id'].split('_', 1)[1]
        sents.add_field('ann_id', [ann_id])

        ann_target = [t for t in target if ann_id in t.get_field('ann_id')]
        # Might be an issue with sentences without corresponding ann targets
        assert len(ann_target) > 0
        sents.add_field('ann_target', ann_target)

        return img, hha, sents, target, self.split_index[idx]

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

        img_id = int(self.split_index[index].split('_')[1])
        img_data = self.coco.imgs[img_id]
        return img_data

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
