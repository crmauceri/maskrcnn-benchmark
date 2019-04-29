from maskrcnn_benchmark.data.datasets.sunspot import ReferExpressionDataset

from nltk.parse.corenlp import CoreNLPDependencyParser
import torch

from allennlp.commands.elmo import ElmoEmbedder, batch_to_ids
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Start the server before running!
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

dependency_parser = CoreNLPDependencyParser(url='http://localhost:9000')

def write_sentence(fp, tokens):
    print(" ".join(tokens))
    fp.write('{}\n'.format(" ".join(tokens)))

def write_subject(fp, tokens):
    parse, = dependency_parser.parse(tokens)
    found = True
    if 'nsubj' in parse.root['deps']:
        nsubj = [parse.nodes[i] for i in parse.root['deps']['nsubj']]
    elif 'nsubjpass' in parse.root['deps']:
        nsubj = [parse.nodes[i] for i in parse.root['deps']['nsubjpass']]
    elif 'NN' in parse.root['tag']:
        nsubj = [parse.root]
    else:
        found = False
    if found:
        for node in nsubj:
            fp.write("('{}','{}')\n".format(node['word'], node['tag']))
    elif not found:
        print(parse.to_conll(4))

# COCO categories for pretty print
CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

if __name__ == "__main__":

    from tqdm import tqdm

    # Run through whole dataset once
    ann_file = '../../datasets/sunspot/annotations/instances.json'
    img_root = '../../datasets/sunspot/images'
    ref_file = '../../datasets/sunspot/annotations/refs(boulder).p'
    vocab_file = '../../datasets/vocab_file.txt'
    refer = ReferExpressionDataset(ann_file, img_root, ref_file, vocab_file, True, active_split="train")

    # with open('../../datasets/sunspot/all_sentences.txt', 'w') as fp:
    #     for id, sent in tqdm(refer.coco.sents.items()):
    #         sent_tokens = sent['tokens']
    #         fp.write('#{}\n'.format(id))
    #         if('.' in sent_tokens and sent_tokens.index('.') != len(sent_tokens)-1):
    #             start = 0
    #             tokens = [i for i, x in enumerate(sent_tokens) if x == "."]
    #             for i in tokens:
    #                 write_sentence(fp, sent_tokens[start:i+1])
    #                 start = i+1
    #         else:
    #             write_sentence(fp, sent_tokens)

    # with open('../../datasets/sunspot/subjects.txt', 'w') as fp:
    #     for id, sent in tqdm(refer.coco.sents.items()):
    #         sent_tokens = sent['tokens']
    #         fp.write('#{}\n'.format(id))
    #         if('.' in sent_tokens and sent_tokens.index('.') != len(sent_tokens)-1):
    #             start = 0
    #             tokens = [i for i, x in enumerate(sent_tokens) if x == "."]
    #             for i in tokens:
    #                 write_subject(fp, sent_tokens[start:i+1])
    #                 start = i+1
    #         else:
    #             write_subject(fp, sent_tokens)

    elmo = ElmoEmbedder()



    for id, sent in tqdm(refer.coco.sents.items()):
         sent_tokens = sent['tokens']
         vectors = elmo.embed_sentence(sent_tokens)
         dist = cosine_similarity(cat_embeddings[0][:,2,0,:], np.expand_dims(vectors[2,2,:], axis=0))
         x = np.argmax(dist)
