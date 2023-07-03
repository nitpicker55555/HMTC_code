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
# torch.set_printoptions(threshold=1000000)  # 设置每行打印的元素数量

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

        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        # print(batch['token'],"batch[token]")
        # print(batch['token'])
        embedding = self.token_embedding(batch['token'])
        
        # print("\n\n")



        # print(batch['token'].to(self.config.train.device_setting.device),"batch['token'].to(self.config.train.device_setting.device)")
        ##print(batch['token'],"batch['token'].to(self.config.train.device_setting.device)")
        #print(batch['token'].shape,"batch.shape")

        # print(embedding,"embedding")
        # get the length of sequences for dynamic rnn, (batch_size, 1)
        seq_len = batch['token_len']
        #print(seq_len,"seqlen")
        #print(seq_len.shape,"seq_len,shape")
        # print(embedding.shape,"embedding.shape")
        # token_output = self.text_encoder(embedding)
        # print(token_output[0].shape)
        # print(token_output,"tokenoutput")
        # print(len(token_output),"token_output.shape")
        # print(token_output[0].shape,"token_output[0].shape")
        # print(token_output[1].shape,"token_output[1].shape")
        # print(token_output[2].shape,"token_output[2].shape")
        token_output_new = embedding[:,0,:]
        print(token_output_new.shape,"token_output_new.shape")
        # print(token_output_new.shape,"token_output_new")
        # logits = self.hiagm(token_output)
        logits = self.hiagm(token_output_new)
        # with open("logits.txt", "w") as file:
        #     # Write the variable to the file
        #     file.write(str(logits))
        #     file.write('\n')
        # # print(logits,"logits")
        # print(logits.shape,"logits shape")
        return logits