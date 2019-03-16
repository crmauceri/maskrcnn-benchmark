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

        # Fix the image ids assigned by the torchvision dataset loader
        # self.index = dict(zip(range(len(self.coco.imgs.keys())), self.coco.imgs.keys()))

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
        file = [file for file in os.listdir(osp.join(self.root, dir, 'HHA')) if file.endswith('png')][0]
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

        # Initialize vocabulary
        with open(vocab_file, 'r') as f:
            self.vocab = [v.strip() for v in f.readlines()]
        self.vocab.extend(['<bos>', '<eos>', '<unk>'])
        self.word2idx = dict(zip(self.vocab, range(1, len(self.vocab) + 1)))

        # Index referring expressions
        self.createRefIndex(ref_file)

        # if dataset == 'refcocog':
        #     self.unique_test_objects = [ref['sent_ids'][0] for key, ref in self.refer.annToRef.items() if
        #                                 ref['split'] == 'val']
        # else:
        #     self.unique_test_objects = [ref['sent_ids'][0] for key, ref in self.refer.annToRef.items() if
        #                                 ref['split'] == 'test']

    def __len__(self):
        return self.length(self.active_split)

    def length(self, split=None):
        if split is None:
            return len(self.index)
        elif split == 'train':
            return len(self.train_index)
        elif split == 'test':
            return len(self.test_index)
        elif split == 'test_unique':
            return len(self.unique_test_objects)
        elif split == 'val':
            return len(self.val_index)

    def __getitem__(self, item):
        return self.getItem(item, self.active_split)

    def getItem(self, idx, split=None):

        if split is None:
            self.ids = self.index #Set coco index
        elif split == 'train':
            self.ids = self.train_index
        elif split == 'test':
            self.ids = self.test_index
        elif split == 'val':
            self.ids = self.val_index

        img, hha, target, img_idx = super().getItem(idx)

        refs = self.coco.imgToRefs[img_idx]

        sentence_t = [s['vocab'] for ref in refs for s in ref['sentences']]
        max_t = max([len(s) for s in sentence_t])
        sentence_t = [[0]*(max_t-len(s)) + s for s in sentence_t]
        sentence_t = torch.as_tensor(sentence_t)
        sents = TensorList(sentence_t)

        refs = [ref for ref in refs if ref['ann_id'] in target.get_field("ann_id")]
        sents.add_field('tokens', [s['tokens'] for ref in refs for s in ref['sentences']])
        sents.add_field('img_id', [s['sent_id'].split('_')[1] for ref in refs for s in ref['sentences']])
        sents.add_field('ann_id', [s['sent_id'].split('_', 1)[1] for ref in refs for s in ref['sentences']])

        # TODO I'm having issues with too little GPU memory, so a temporary fix...
        # Randomly choose a sentence
        sents = sents[randint(0, len(sents)-1)]

        return img, hha, sents, target, img_idx

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
                self.sent2vocab(sent)
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

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

        self.train_index = list(set([self.coco.refs[ref]['image_id'] for ref in self.coco.refs if self.coco.refs[ref]['split'] == 'train']))
        self.train_index.sort()

        self.val_index = list(set([self.coco.refs[ref]['image_id'] for ref in self.coco.refs if self.coco.refs[ref]['split'] == 'val']))
        self.val_index.sort()

        self.test_index = list(set([self.coco.refs[ref]['image_id'] for ref in self.coco.refs if self.coco.refs[ref]['split'] == 'test']))
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
