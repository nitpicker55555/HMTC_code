#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from train_modules.evaluation_metrics import evaluate
import torch
import tqdm
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


class Trainer(object):
    def __init__(self, model, criterion, optimizer, vocab, config, global_prototype_tensor, ac_matrix):
        """
        :param model: Computational Graph
        :param criterion: train_modules.ClassificationLoss object
        :param optimizer: optimization function for backward pass
        :param vocab: vocab.v2i -> Dict{'token': Dict{vocabulary to id map}, 'label': Dict{vocabulary
        to id map}}, vocab.i2v -> Dict{'token': Dict{id to vocabulary map}, 'label': Dict{id to vocabulary map}}
        :param config: helper.Configure object
        """
        super(Trainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.config = config
        self.criterion = criterion
        self.device = self.config.train.device_setting.device
        self.optimizer = optimizer
        self.global_prototype_tensor = global_prototype_tensor
        self.ac_matrix = ac_matrix
        # self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.proto_model.parameters()),
        # lr=self.config.train.optimizer.learning_rate)

    def update_lr(self):
        """
        (callback function) update learning rate according to the decay weight
        """
        logger.warning('Learning rate update {}--->{}'
                       .format(self.optimizer.param_groups[0]['lr'],
                               self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def run(self, data_loader, epoch, stage, mode='TRAIN'):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        predict_probs = []
        target_labels = []
        total_loss = 0.0
        num_batch = data_loader.__len__()
        global memory_bank,memory_bank_size,memory_labels
        memory_bank=None
        memory_bank_size=200

        def labelContrastiveMask(labels, memory_label=None):
            labels = torch.cat([labels, memory_label])
            labels_reshaped = labels.view(-1, 1)
            #  broadcasting to calculate equal or not
            mask = labels_reshaped == labels_reshaped.t()
            #  bool tensor to float tensor
            mask = mask.float()
            positive_mask = mask
            return positive_mask
        def memory_generate(batch_examples,labels):
            global memory_bank,memory_labels
            if memory_bank == None:
                memory_bank = batch_examples
                memory_labels=labels
            else:
                memory_bank = torch.cat([memory_bank, batch_examples.detach()], dim=0)
                memory_labels=torch.cat([memory_labels, labels.detach()], dim=0)
                if memory_bank.size()[0] > memory_bank_size:
                    memory_bank = memory_bank[-memory_bank_size:, :]
                    memory_labels=memory_labels[-memory_bank_size:, :]

        for batch in tqdm.tqdm(data_loader):
            logits, label_information = self.model(batch) #label_information 64 103 768
            """
            double each label to positive one and negative one 
            example:
            label_list = [[1, 2], [0, 1], [0, 2]]
            positive_mask:
            tensor([[0., 0., 1., 0., 1., 0.],
                    [1., 0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 1., 0.]])
            """
            positive_mask = torch.zeros((label_information.shape[0], label_information.shape[1] * 2), dtype=torch.int).to(
                self.device)   #generate doubled labels

            for i, sample_labels in enumerate(batch['label_list']):
                for label in sample_labels:
                    # set each label positive
                    positive_mask[i, label * 2] = 1
                    # else will be 0
            memory_generate(label_information,positive_mask)
            label_loss_pos_mask = labelContrastiveMask(positive_mask,memory_labels )


            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.hiagm.linear.weight
            else:
                recursive_constrained_params = None
            classification_loss = self.criterion(logits,
                                                 batch['label'].to(self.config.train.device_setting.device),
                                                 recursive_constrained_params)

            # calculate contrastive loss according to different mask
            label_loss = self.SampleContrastiveloss(label_information.to(self.device), label_loss_pos_mask.to(self.device), torch.tensor(self.ac_matrix).to(
                                                                                self.device),memory_bank.to(
                                                                                self.device))
            total_loss = label_loss  + classification_loss

            # print(total_loss, "total_loss")
            if mode == 'TRAIN':
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])
        total_loss = total_loss / num_batch
        if mode == 'EVAL':
            metrics = evaluate(predict_probs,
                               target_labels,
                               self.vocab,
                               self.config.eval.threshold)
            # metrics = {'precision': precision_micro,
            #             'recall': recall_micro,
            #             'micro_f1': micro_f1,
            #             'macro_f1': macro_f1}
            logger.info("%s performance at epoch %d --- Precision: %f, "
                        "Recall: %f, Micro-F1: %f, Macro-F1: %f, Loss: %f.\n"
                        % (stage, epoch,
                           metrics['precision'], metrics['recall'], metrics['micro_f1'], metrics['macro_f1'],
                           total_loss))
            return metrics

    def train(self, data_loader, epoch):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, 'Train', mode='TRAIN')

    def eval(self, data_loader, epoch, stage):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, TRAIN/DEV/TEST, log the result of the according corpus
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode='EVAL')

    def SampleContrastiveloss(self,features, positive_mask, cost_matrix, memory_features=None, temperature=0.07):

        # normalize the input tensor along dimension 1
        features = torch.cat([features, memory_features]).detach()
        batch_size, num_labels, _ = features.shape
        features = F.normalize(features, dim=2)
        features = features.view(batch_size * num_labels, -1)
        N = (batch_size ** 2) * num_labels
        logits_mask = torch.ones_like(positive_mask).to(self.device)
        self_contrast_mask = 1 - torch.diag(torch.ones((positive_mask.size()[0]))).to(self.device)
        logits_mask[:, :positive_mask.size()[0]] = logits_mask[:, :positive_mask.size()[0]].clone() * self_contrast_mask
        positive_mask = positive_mask * logits_mask
        # print(positive_mask.shape, "positive_mask")
        # reshape features to (batch_size * num_labels, feature_size)
        s = features
        s_norm = F.normalize(s, p=2, dim=1)
        d = 1. / (1. + torch.exp(torch.mm(s_norm, s_norm.t())))  # batch_size*num_labels
        # print(d.shape, "d")
        exp_s = torch.exp(positive_mask * d)
        cost_matrix = cost_matrix.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)# repeat_interleave double cost_matrix
        cost_matrix = cost_matrix.repeat(batch_size, batch_size)  # repeat cost_matrix to macth batch_size*num_labels
        exp_mask = torch.exp((1. - positive_mask) * d * cost_matrix)  # apply cost_matrix on negative pairs
        sum_exp_mask = torch.sum(exp_mask, dim=1, keepdim=True)  # log added here

        pos_loss = - torch.log(torch.sum(exp_s / sum_exp_mask) / N)
        return (pos_loss)

