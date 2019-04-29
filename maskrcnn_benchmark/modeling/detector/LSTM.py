import torch
import torch.nn as nn

#torch.manual_seed(1)

from maskrcnn_benchmark.structures.tensorlist import to_tensor_list

from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

## Stand alone model instructions
#
# Use tools/train_net.py to train
# LanguageModel use configs/LSTM_language_experiment.yaml
# ClassificationModel use configs/LSTM_classification_experiment.yaml

## TODO
# - Check loading from checkpoint
# - Integrate with refexp_localizer
# - Testing
# - Check if gradients are cleared at start of each iteration

#Network Definition
class LanguageModel(nn.Module):

    def __init__(self, cfg, class_dim=None, embed_dim=None):
        super(LanguageModel, self).__init__()

        self.total_loss = []
        self.val_loss = []
        self.start_epoch = 0
        self.loss_function = SequenceLoss(nn.CrossEntropyLoss(), cfg.LOSS_WEIGHTS.TEXT_LOSS)

        self.vocab_dim = cfg.MODEL.LSTM.VOCAB_N+1
        if class_dim:
            self.class_dim = class_dim
        else:
            self.class_dim = cfg.MODEL.LSTM.VOCAB_N+1
        self.hidden_dim = cfg.MODEL.LSTM.HIDDEN
        self.dropout_p = cfg.MODEL.LSTM.DROPOUT
        self.feats_dim = cfg.MODEL.LSTM.ADDITIONAL_FEATS

        #Word Embeddings
        if embed_dim:
            self.embed_dim = embed_dim
        else:
            self.embed_dim = self.hidden_dim
        self.embedding = torch.nn.Embedding(self.vocab_dim, self.embed_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.dropout1 = nn.Dropout(p=self.dropout_p)
        self.lstm = nn.LSTM(self.embed_dim + self.feats_dim, self.hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(p=self.dropout_p)
        self.hidden2class = nn.Linear(self.hidden_dim, self.class_dim)
        self.hidden = self.init_hidden(1)


    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim, device=next(self.parameters()).device, requires_grad=True),
                torch.zeros(1, batch_size, self.hidden_dim, device=next(self.parameters()).device, requires_grad=True))

    @staticmethod
    def get_checkpt_file(checkpt_file, hidden_dim, feats_dim, dropout_p):
        return '{}_hidden{}_feats{}_dropout{}.mdl'.format(checkpt_file, hidden_dim, feats_dim, dropout_p)

    def checkpt_file(self, checkpt_file):
        return self.get_checkpt_file(checkpt_file, self.hidden_dim, self.feats_dim, self.dropout_p)

    def prepare_instance(self, ref, device):
        instances = to_tensor_list(ref[2])
        targets = instances[:, 1:]
        instances = instances[:, :-1]

        instances = instances.to(device)
        targets = targets.to(device)
        return instances, targets

    def forward(self, ref, device, targets=None):
        instance, targets = self.prepare_instance(ref, device)
        self.clear_gradients(len(instance))

        embeds = self.embedding(instance.tensor)
        embeds = self.dropout1(embeds)
        batch_size, seq_len, feat_dim = embeds.size()

        if instance.has_field('feats'):
            feats = instance['feats'].repeat(seq_len, 1, 1).permute(1, 0, 2)

            # Concatenate text embedding and additional features
            # TODO fix for Maoetal_Full
            if embeds.size()[0] == 1:
                embeds = torch.cat([embeds.repeat(feats.size()[0], 1, 1), feats], 2)
            else:
                embeds = torch.cat([embeds, feats], 2)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = self.dropout2(lstm_out)
        class_space = self.hidden2class(lstm_out)

        if self.training:
            losses = {'seq_loss': self.loss_function(class_space, targets)}
            return losses
        else:
            return class_space

    def make_ref(self, word_idx, feats=None):
        ref = {'vocab_tensor': torch.tensor([word_idx, -1], dtype=torch.long, device=self.device).unsqueeze(0)}
        if feats is not None:
            ref['feats'] = feats
        return ref

    def clear_gradients(self, batch_size):
        # self.zero_grad()
        self.hidden = self.init_hidden(batch_size)

    def infer(self, instance, feats=None):
        sentence = []
        start_word = '<bos>'
        word_idx = self.word2idx[start_word]
        end_idx = self.word2idx['<eos>']

        with torch.no_grad():
            self.hidden = self.init_hidden(1)

            while word_idx != end_idx and len(sentence) < 30:
                ref = self.make_ref(word_idx, feats)
                output = self(ref)
                word_idx = torch.argmax(output)
                sentence.append(self.ind2word[word_idx])

        return sentence

class ClassificationModel(LanguageModel):

    def __init__(self, cfg):
        super(ClassificationModel, self).__init__(cfg, class_dim=cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, embed_dim=1024)

        #Word Embeddings
        self.embedding = Elmo(options_file, weight_file, 2, dropout=0)

        #Loss Function
        self.loss_function = ClassificationLoss(nn.CrossEntropyLoss(), cfg.LOSS_WEIGHTS.TEXT_LOSS)

    def prepare_instance(self, ref, device):
        instances = to_tensor_list(ref[2])
        targets = torch.tensor([a.get_field('labels') for a in instances.get_field('ann_target')])

        instances = instances.to(device)
        targets = targets.to(device)
        return instances, targets

    def forward(self, ref, device, targets=None):
        instance, targets = self.prepare_instance(ref, device)
        self.clear_gradients(len(instance))

        sentence = instance.get_field('tokens')
        sentence_ids = batch_to_ids(sentence)
        embeds = self.embedding(sentence_ids)
        embeds = self.dropout1(embeds['elmo_representations'][1])
        batch_size, seq_len, feat_dim = embeds.size()

        if instance.has_field('feats'):
            assert(False, 'Untested code')
            feats = instance['feats'].repeat(seq_len, 1, 1).permute(1, 0, 2)

            #Concatenate text embedding and additional features
            #TODO fix for Maoetal_Full
            if embeds.size()[0]==1:
                embeds = torch.cat([embeds.repeat(feats.size()[0], 1, 1), feats], 2)
            else:
                embeds = torch.cat([embeds, feats], 2)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = self.dropout2(lstm_out)
        class_space = self.hidden2class(lstm_out)

        if self.training:
            losses = {'seq_loss': self.loss_function(class_space, targets)}
            return losses
        else:
            return class_space

class SequenceLoss(nn.Module):
    def __init__(self, loss_function, loss_weight = 1.0):
        super(SequenceLoss, self).__init__()
        self.Loss = loss_function
        self.loss_weight = loss_weight

    def forward(self, embeddings, targets, per_instance=False):
        target_tensor = targets.tensor
        if per_instance:
            loss = torch.zeros(embeddings.size()[0], device=self.device)
            for step in range(target_tensor.size()[1]):
                loss += self.Loss(embeddings[:, step, :], target_tensor[:, step])
        else:
            loss = 0.0
            for step in range(target_tensor.size()[1]):
                loss += self.Loss(embeddings[:, step, :], target_tensor[:, step])

        loss *= self.loss_weight
        return loss


class ClassificationLoss(nn.Module):
    def __init__(self, loss_function, loss_weight = 1.0):
        super(ClassificationLoss, self).__init__()
        self.Loss = loss_function
        self.loss_weight = loss_weight

    def forward(self, embeddings, target_class, per_instance=False):
        if per_instance:
            loss = torch.zeros(embeddings.size()[0], device=self.device)
            loss += self.Loss(embeddings[:, -1, :], target_class)
        else:
            loss = self.Loss(embeddings[:, -1, :], target_class)

        loss *= self.loss_weight
        return loss
