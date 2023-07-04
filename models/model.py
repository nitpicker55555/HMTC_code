#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
from models.structure_model.structure_encoder import StructureEncoder
# from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.multi_label_attention import HiAGMLA
from models.text_feature_propagation import HiAGMTP
from models.origin import Classifier
import torch
from models.Graph_encoder.graph import GraphEncoder
# torch.set_printoptions(threshold=1000000)  # 设置每行打印的元素数量
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import BertModel
import os
DATAFLOW_TYPE = {
    'HiAGM-TP': 'serial',
    'HiAGM-LA': 'parallel',
    'Origin': 'origin'
}


class HiAGM(nn.Module):
    def __init__(self, config, vocab, model_type, model_mode='TRAIN'):
        """
        Hierarchy-Aware Global Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_type: Str, ('HiAGM-TP' for the serial variant of text propagation,
                                 'HiAGM-LA' for the parallel variant of multi-label soft attention,
                                 'Origin' without hierarchy-aware module)
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(HiAGM, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device
        self.pretrain_config = BertModel.from_pretrained('bert-base-uncased')
        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']

        self.token_embedding = EmbeddingLayer(
            vocab_map=self.token_map,
            embedding_dim=config.embedding.token.dimension,
            vocab_name='token',
            config=config,
            padding_index=vocab.padding_index,
            pretrained_dir=config.embedding.token.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.token.init_type
        )

        self.dataflow_type = DATAFLOW_TYPE[model_type]

        # self.text_encoder = TextEncoder(config)
        self.structure_encoder = StructureEncoder(config=config,
                                                  label_map=vocab.v2i['label'],
                                                  device=self.device,
                                                  graph_model_type=config.structure_encoder.type)

        if self.dataflow_type == 'serial':
            self.hiagm = HiAGMTP(config=config,
                                 device=self.device,
                                 graph_model=self.structure_encoder,
                                 label_map=self.label_map)
        elif self.dataflow_type == 'parallel':
            self.hiagm = HiAGMLA(config=config,
                                 device=self.device,
                                 graph_model=self.structure_encoder,
                                 label_map=self.label_map,
                                 model_mode=model_mode)
        else:
            self.hiagm = Classifier(config=config,
                                    vocab=vocab,
                                    device=self.device)

    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        # params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.hiagm.parameters()})
        return params

    def forward(self, batch):
        """
        forward pass of the overall architecture
        :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return:
        """


        embedding = self.token_embedding(batch['token']).to(self.device)


        token_output_new = embedding[:, 0, :].to(self.device)

        inputs_embeds = token_output_new.unsqueeze(1).to(self.device)
        print(inputs_embeds.shape,"inputs_embeds")
        attention_mask = torch.ones(token_output_new.shape[0], 1).to(self.device)
        print(attention_mask.shape,"attention_mask")
        labels = batch['label'].to(self.device)
        print(labels.shape,"labels")



        model = GraphEncoder(config=self.pretrain_config.config, graph=False, layer=1, data_path=os.getcwd()+"/data/rcv1",
                             threshold=0.01, tau=1,)

        model_bert = BertModel(self.pretrain_config.config).to(self.device)
        output = model.forward(inputs_embeds, attention_mask, labels, lambda x: model_bert.embeddings(x.to(self.device))[0].to(self.device))
        print(output.shape,"output")
        logits = self.hiagm(token_output_new).to(self.device)

        return logits