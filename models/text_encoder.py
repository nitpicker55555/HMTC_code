#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    def __init__(self, config):
        """
        TextRCNN
        :param config: helper.configure, Configure Object
        """
        super(TextEncoder, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.device = config.train.device_setting.device
        hidden_dimension = self.bert.config.hidden_size
        self.kernel_sizes = config.text_encoder.CNN.kernel_size
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                hidden_dimension,
                config.text_encoder.CNN.num_kernel,
                kernel_size,
                padding=kernel_size // 2
                )
            )
        self.top_k = config.text_encoder.topK_max_pooling
        self.dropout = torch.nn.Dropout(p=config.text_encoder.RNN.dropout)

    def forward(self, inputs):
        """
        :param inputs: torch.LongTensor, token ids, (batch, max_len)
        :return:
        """
        inputs = inputs.to(self.device)
        # text_output = self.dropout(inputs)
        text_output=inputs
        # print(text_output.shape,"text_output.shape")
        text_output = text_output.transpose(1, 2).to(self.device)
        # print(text_output.shape,"text_output.shape_bert")
        topk_text_outputs = []
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(text_output)).to(self.device)
            topk_text = torch.topk(convolution, self.top_k)[0].view(text_output.size(0), -1).to(self.device)
            topk_text = topk_text.unsqueeze(1).to(self.device)
            topk_text_outputs.append(topk_text).to(self.device)
        return topk_text_outputs