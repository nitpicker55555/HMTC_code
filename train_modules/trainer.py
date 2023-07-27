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

        def labelContrastiveMask(labels,distance_matrix):
            #treat each embedding with same label and label equal to 1 as positive
            #labels=tensor([[1, 1],[1, 0]]) shape with num_labels，batch_size
            #mask=tensor([[0,0,1,0],[0,0,0,0],[1,0,0,0],[0,0,0,0]] shape with num_labels*batch_size,num_labels*batch_size
            num_labels, batch_size = labels.size()
            mask = torch.zeros(num_labels * batch_size, num_labels * batch_size, dtype=torch.float32)

            for i in range(batch_size):
                mask[i::batch_size, i::batch_size] = torch.outer(labels[:, i], labels[:, i])
            positive_mask=mask
            # reverse positive_mask to get negative_mask
            negative_mask=~mask
            # repeat the distance matrix to match the dimension of mask and generate weighted mask
            distance_matrix = distance_matrix.repeat(mask.shape[0],mask.shape[1])
            positive_mask=distance_matrix*positive_mask
            negative_mask=distance_matrix*negative_mask
            return positive_mask,negative_mask

        def sampleContrastiveMask(labels,distance_matrix):

            #for each sample treat each positive label's embedding as positive
            #labels=tensor([[1, 1],[1, 0]]) shape with num_labels，batch_size
            #mask=tensor([[0,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]] shape with num_labels*batch_size,num_labels*batch_size

            num_labels, batch_size = labels.size()
            mask = torch.zeros(num_labels * batch_size, num_labels * batch_size, dtype=torch.float32)

            # reshape labels from shape (num_labels, batch_size) to (num_labels * batch_size)
            labels = labels.view(-1)

            for i in range(num_labels):
                # get the start and end index for each block of size 'num_labels' in the mask
                start_idx, end_idx = i * batch_size, (i + 1) * batch_size

                # set mask value to 1 only when both labels values are 1
                mask[start_idx:end_idx, start_idx:end_idx] = torch.outer(labels[start_idx:end_idx],
                                                                         labels[start_idx:end_idx])
            positive_mask=mask
            # reverse positive_mask to get negative_mask
            negative_mask=~mask
            #repeat the distance matrix to match the dimension of mask and generate weighted mask
            distance_matrix = distance_matrix.repeat(mask.shape[0],mask.shape[1])
            positive_mask=distance_matrix*positive_mask
            negative_mask=distance_matrix*negative_mask
            return positive_mask,negative_mask

        for batch in tqdm.tqdm(data_loader):
            logits, label_information = self.model(batch)
            positive_mask = torch.zeros((label_information.shape[0], label_information.shape[1]), dtype=torch.int)
            for idx, labels in enumerate( batch['label_list']):
                positive_mask[idx, labels] = 1
            label_loss_pos_mask,label_loss_neg_mask=labelContrastiveMask(positive_mask,torch.tensor(self.ac_matrix).to(self.device) )
            sample_loss_pos_mask,sample_loss_neg_mask=sampleContrastiveMask(positive_mask,torch.tensor(self.ac_matrix).to(self.device) )
            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.hiagm.linear.weight
            else:
                recursive_constrained_params = None
            classification_loss = self.criterion(logits,
                                                 batch['label'].to(self.config.train.device_setting.device),
                                                 recursive_constrained_params)

            # calculate contrastive loss according to different mask
            label_loss=self.LabelContrastiveloss(label_information,label_loss_pos_mask,label_loss_neg_mask)
            sample_loss = self.SampleContrastiveloss(label_information, sample_loss_pos_mask,sample_loss_neg_mask)



            # calculate total loss
            total_loss = label_loss+sample_loss+classification_loss
            print(total_loss, "total_loss")
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
    def SampleContrastiveloss(self, features, positive_mask,negative_mask,temperature=0.07):
        features = F.normalize(features, dim=2)

        batch_size, num_labels, _ = features.shape


        # reshape features to (batch_size * num_labels, feature_size)
        features = features.view(batch_size * num_labels, -1)

        # compute all pair-wise similarities
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(logits),
            1,
            torch.arange(batch_size * num_labels).view(-1, 1).to(self.device),
            0
        )

        # compute log_prob

        positive_mask = positive_mask * logits_mask
        negative_mask=negative_mask*logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        mean_log_prob_neg = (negative_mask * log_prob).sum(1) / negative_mask.sum(1)

        # loss
        loss = - 0.5 * (mean_log_prob_pos + mean_log_prob_neg)
        loss = loss.nanmean()
        return loss


    def LabelContrastiveloss(self, features, positive_mask,negative_mask,temperature=0.07):
        features = F.normalize(features, dim=2)

        batch_size, num_labels, _ = features.shape


        # reshape features to (batch_size * num_labels, feature_size)
        features = features.view(batch_size * num_labels, -1)

        # compute all pair-wise similarities
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(logits),
            1,
            torch.arange(batch_size * num_labels).view(-1, 1).to(self.device),
            0
        )

        # compute log_prob

        positive_mask = positive_mask * logits_mask
        negative_mask=negative_mask*logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        mean_log_prob_neg = (negative_mask * log_prob).sum(1) / negative_mask.sum(1)

        # loss
        loss = - 0.5 * (mean_log_prob_pos + mean_log_prob_neg)
        loss = loss.nanmean()

        return loss
