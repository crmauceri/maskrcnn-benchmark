import torch
import torchvision

import os.path as osp
import json
import pickle

from maskrcnn_benchmark.data.datasets.coco import COCODataset



class ReferExpressionDataset(COCODataset):
    def __init__(
        self, ann_file, img_root, ref_file, vocab_file, remove_images_without_annotations, \
            transforms=None, disable_cuda=False, active_split=None
    ):
        super().__init__(ann_file, img_root, remove_images_without_annotations, transforms)

        # Fix the image ids assigned by the torchvision dataset loader
        self.ids = dict(zip(self.coco.imgs.keys(), self.coco.imgs.keys()))

        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.active_split = active_split

        with open(vocab_file, 'r') as f:
            self.vocab = [v.strip() for v in f.readlines()]


        self.vocab.extend(['<bos>', '<eos>', '<unk>'])
        self.word2idx = dict(zip(self.vocab, range(1, len(self.vocab) + 1)))

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
            img_idx = self.index[idx]
        elif split == 'train':
            img_idx = self.train_index[idx]
        elif split == 'test':
            img_idx = self.test_index[idx]
        # elif split == 'test_unique':
        #     img_idx = self.unique_test_objects[idx]
        elif split == 'val':
            img_idx = self.val_index[idx]

        img, target, idx = super().__getitem__(img_idx)
        refs = self.coco.imgToRefs[img_idx]
        sents = [ref['sentences'] for ref in refs]

        return img, target, sents

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

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = self.coco.anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                self.sent2vocab(sent)
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

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

        for sent in self.coco.sents:
            padding = [0.0] * (self.max_sent_len - len(self.coco.sents[sent]['vocab']))
            self.coco.sents[sent]['vocab_tensor'] = torch.tensor(padding + self.coco.sents[sent]['vocab'], dtype=torch.long,
                                                device=self.device)

        self.index = list(set([self.coco.refs[ref]['image_id'] for ref in self.coco.refs]))
        self.train_index = list(set([self.coco.refs[ref]['image_id'] for ref in self.coco.refs if self.coco.refs[ref]['split'] == 'train']))
        self.val_index = list(set([self.coco.refs[ref]['image_id'] for ref in self.coco.refs if self.coco.refs[ref]['split'] == 'val']))
        self.test_index = list(set([self.coco.refs[ref]['image_id'] for ref in self.coco.refs if self.coco.refs[ref]['split'] == 'test']))

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