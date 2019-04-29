# from tqdm import *
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# #torch.manual_seed(1)
#
# DEBUG = False
#
# class Classifier(nn.Module):
#     def __init__(self, loss_weight = 1.0):
#         super(Classifier, self).__init__()
#         self.total_loss = []
#         self.val_loss = []
#         self.start_epoch = 0
#         self.loss_function = SequenceLoss(nn.CrossEntropyLoss(), loss_weight)
#
#     def forward(self, instance):
#         pass
#
#     def load_model(self, checkpt_file):
#         print("=> loading checkpoint '{}'".format(checkpt_file))
#         checkpoint = torch.load(checkpt_file, map_location=lambda storage, loc: storage)
#
#         self.start_epoch = checkpoint['epoch']
#         self.total_loss = checkpoint['total_loss']
#         self.val_loss = checkpoint['val_loss']
#         self.load_state_dict(checkpoint['state_dict'])
#         self.load_params(checkpoint)
#
#         print("=> loaded checkpoint '{}' (epoch {})"
#               .format(checkpt_file, checkpoint['epoch']))
#
#     def load_params(self, checkpoint):
#         pass
#
#     def save_model(self, checkpt_prefix, params):
#         print("=> saving checkpoint '{}'".format(self.checkpt_file(checkpt_prefix)))
#         torch.save(params, self.checkpt_file(checkpt_prefix))
#
#     def checkpt_file(self, checkpt_prefix):
#         return '{}.mdl'.format(checkpt_prefix)
#
#     def run_training(self, cfg, dataloader):
#         checkpt_prefix = cfg.OUTPUT_DIR
#
#         n_epochs = cfg.SOLVER.MAX_ITER
#         learning_rate = cfg.SOLVER.BASE_LR
#         batch_size = cfg.SOLVER.IMS_PER_BATCH
#         l2_reg_fraction = cfg.SOLVER.L2_REG_FRACTION
#         optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, weight_decay=l2_reg_fraction)
#
#         for epoch in range(self.start_epoch, n_epochs):
#             self.train()
#             self.total_loss.append(0)
#
#             for i_batch, sample_batched in enumerate(dataloader): #, desc='{}rd epoch'.format(epoch)):
#
#                 instances, targets = self.prepare_instance(sample_batched)
#                 self.clear_gradients(batch_size=len(instances))
#
#                 loss = 0
#                 label_scores = self(instances)
#                 loss += self.loss_function(label_scores, targets)
#
#                 if DEBUG:
#                     print([self.wordnet.ind2word[instances['vocab_tensor'][0, i]] for i in range(instances['vocab_tensor'].size()[1])])
#                     print([self.wordnet.ind2word[torch.argmax(label_scores[0, i, :])] for i in range(instances['vocab_tensor'].size()[1]-1)])
#                     print(loss)
#
#                 loss.backward()
#                 optimizer.step()
#
#                 if DEBUG:
#                     self.clear_gradients(batch_size=1)
#                     print(self.infer('<bos>', feats=instances['feats'][0]))
#
#                 self.total_loss[epoch] += loss.item()
#
#             self.total_loss[epoch] = self.total_loss[epoch] / float(i_batch)
#
#             self.clear_gradients(batch_size=1)
#             self.save_model(checkpt_prefix, {
#                 'epoch': epoch + 1,
#                 'state_dict': self.state_dict(),
#                 'total_loss': self.total_loss,
#                 'val_loss': self.val_loss})
#
#             print('Average training loss:{}'.format(self.total_loss[epoch]))
#
#             if epoch % 2 == 0:
#                 self.save_model('{}.checkpoint{}'.format(checkpt_prefix, epoch), {
#                     'epoch': epoch,
#                     'state_dict': self.state_dict(),
#                     'total_loss': self.total_loss,
#                     'val_loss': self.val_loss})
#
#                 self.val_loss.append(0)
#                 self.val_loss[-1] = self.run_testing(dataloader)
#                 print('Average validation loss:{}'.format(self.total_loss[epoch]))
#
#         return self.total_loss
#
#     def prepare_instance(self, ref):
#         pass
#
#     def run_testing(self, dataloader):
#         self.eval()
#
#         total_loss = 0
#         for k, instance in enumerate(tqdm(dataloader, desc='Validation')):
#             with torch.no_grad():
#                 instances, targets = self.prepare_instance(instance)
#                 self.clear_gradients(batch_size=len(instance))
#
#                 label_scores = self(instances)
#                 total_loss += self.loss_function(label_scores, targets)
#         return total_loss/float(k)
#
#     def run_inference(self, dataloader):
#         self.eval()
#
#         generated_exp = [0]*len(dataloader)
#         for k, instance in enumerate(tqdm(dataloader, desc='Generation')):
#             instances, targets = self.prepare_instance(instance)
#             generated_exp[k] = dict()
#             generated_exp[k]['inference'] = ' '.join(self.infer("<bos>", instances))
#             generated_exp[k]['refID'] = instance['refID'].item()
#             generated_exp[k]['imgID'] = instance['imageID'].item()
#             generated_exp[k]['objID'] = instance['objectID'][0]
#             generated_exp[k]['objClass'] = instance['objectClass'][0]
#
#         return generated_exp
#
#     def infer(self, instance, feats=None):
#         with torch.no_grad():
#             return self(instance)
#
#     def clear_gradients(self, batch_size=None):
#         self.zero_grad()
#
#
# class SequenceLoss(nn.Module):
#     def __init__(self, loss_function, loss_weight = 1.0):
#         super(SequenceLoss, self).__init__()
#         self.Loss = loss_function
#         self.loss_weight = loss_weight
#
#     def forward(self, embeddings, targets, per_instance=False):
#         target_tensor = targets.tensor
#         if per_instance:
#             loss = torch.zeros(embeddings.size()[0], device=self.device)
#             for step in range(target_tensor.size()[1]):
#                 loss += self.Loss(embeddings[:, step, :], target_tensor[:, step])
#         else:
#             loss = 0.0
#             for step in range(target_tensor.size()[1]):
#                 loss += self.Loss(embeddings[:, step, :], target_tensor[:, step])
#
#         loss *= self.loss_weight
#         return loss
#
#
# class ClassificationLoss(nn.Module):
#     def __init__(self, loss_function, loss_weight = 1.0):
#         super(ClassificationLoss, self).__init__()
#         self.Loss = loss_function
#         self.loss_weight = loss_weight
#
#     def forward(self, embeddings, target_class, per_instance=False):
#         if per_instance:
#             loss = torch.zeros(embeddings.size()[0], device=self.device)
#             loss += self.Loss(embeddings[:, -1, :], target_class)
#         else:
#             loss = self.Loss(embeddings[:, -1, :], target_class)
#
#         loss *= self.loss_weight
#         return loss