#!/usr/bin/env python
# coding:utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import helper.logger as logger
from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_
from transformers import BertModel, BertTokenizer
import logging
torch.set_printoptions(threshold=1000000000000000000000000)  # 设置每行打印的元素数量
logging.getLogger("transformers").setLevel(logging.ERROR)
INIT_FUNC = {
    'uniform': uniform_,
    'kaiming_uniform': kaiming_uniform_,
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
    'kaiming_normal': kaiming_normal_
}


class EmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 vocab_map,
                 embedding_dim,
                 vocab_name,
                 config,
                 padding_index=None,
                 pretrained_dir=None,
                 model_mode='TRAIN',
                 initial_type='kaiming_uniform',
                 negative_slope=0, mode_fan='fan_in',
                 activation_type='linear',
                 ):
        """
        embedding layer
        :param vocab_map: vocab.v2i[filed] -> Dict{Str: Int}
        :param embedding_dim: Int, config.embedding.token.dimension
        :param vocab_name: Str, 'token' or 'label'
        :param config: helper.configure, Configure Object
        :param padding_index: Int, index of padding word
        :param pretrained_dir: Str,  file path for the pretrained embedding file
        :param model_mode: Str, 'TRAIN' or 'EVAL', for initialization
        :param initial_type: Str, initialization type
        :param negative_slope: initialization config
        :param mode_fan: initialization config
        :param activation_type: None
        """
        super(EmbeddingLayer, self).__init__()
        self.vocab_map = vocab_map
        self.dropout = torch.nn.Dropout(p=config['embedding'][vocab_name]['dropout'])
        # self.embedding = torch.nn.Embedding(len(vocab_map), embedding_dim, padding_index)
        self.device = config.train.device_setting.device
        # initialize lookup table
        assert initial_type in INIT_FUNC
        if initial_type.startswith('kaiming'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=negative_slope,
                                                        mode=mode_fan,
                                                        nonlinearity=activation_type)
        elif initial_type.startswith('xavier'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        gain=torch.nn.init.calculate_gain(activation_type))
        else:
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=-0.25,
                                                        b=0.25)

        # if model_mode == 'TRAIN' and config['embedding'][vocab_name]['type'] == 'pretrain' \
        #         and pretrained_dir is not None and pretrained_dir != '':
        #     self.load_pretrained(embedding_dim, vocab_map, vocab_name, pretrained_dir)

        # if padding_index is not None:
        #     self.lookup_table[padding_index] = 0.0
        # self.embedding.weight.data.copy_(self.lookup_table)
        # self.embedding.weight.requires_grad = True
        # del self.lookup_table

    def load_pretrained(self, embedding_dim, vocab_map, vocab_name, pretrained_dir):
        """
        load pretrained file
        :param embedding_dim: Int, configure.embedding.field.dimension
        :param vocab_map: vocab.v2i[field] -> Dict{v:id}
        :param vocab_name: field
        :param pretrained_dir: str, file path
        """
        # logger.info('Loading {}-dimension {} embedding from pretrained file: {}'.format(
        #     embedding_dim, vocab_name, pretrained_dir))
        # with open(pretrained_dir, 'r', encoding='utf8') as f_in:
        #     num_pretrained_vocab = 0
        #     for line in f_in:
        #         row = line.rstrip('\n').split(' ')
        #         if len(row) == 2:
        #             assert int(row[1]) == embedding_dim, 'Pretrained dimension %d dismatch the setting %d' \
        #                                                  % (int(row[1]), embedding_dim)
        #             continue
        #         if row[0] in vocab_map:
        #             current_embedding = torch.FloatTensor([float(i) for i in row[1:]])
        #             self.lookup_table[vocab_map[row[0]]] = current_embedding
        #             num_pretrained_vocab += 1
        # logger.info('Total vocab size of %s is %d.' % (vocab_name, len(vocab_map)))
        # logger.info('Pretrained vocab embedding has %d / %d' % (num_pretrained_vocab, len(vocab_map)))

    def forward(self, vocab_id_list):
        """
        :param vocab_id_list: torch.Tensor, (batch_size, max_length)
        :return: embedding -> torch.FloatTensor, (batch_size, max_length, embedding_dim)
        """

        # 初始化一个线性层，输入大小是768（BERT的输出），输出大小是300
        # linear = nn.Linear(768, 300)
        model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        label_dict = []

        with open(os.getcwd() + '/vocab/label.dict', 'r', encoding='utf-8') as file:
            for index, line in enumerate(file):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    key = str(index)
                    value = parts[0]
                    label_dict.append(value)

        # 避免模型在前向传播时计算不必要的梯度
        model.eval()

        word_embeddings = []
        # print(vocab_id_list, "vocab_id_list")
        # print(len(vocab_id_list), "len(vocab_id_list)")
        
        if isinstance(vocab_id_list[0], list):

            for sentence_id in vocab_id_list:
                marked_sentence = sentence_id

                # 使用tokenizer将tokens转为input_ids
                input_ids = tokenizer.convert_tokens_to_ids(marked_sentence)

                # 将input_ids转为Tensor，并添加一个新的维度以符合BERT的输入需求
                input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)

                # 获取BERT模型的输出
                outputs = model(input_tensor)

                # 获取每一个token的隐藏状态
                sentence_embeddings = outputs[0]
                # print(sentence_embeddings.device,"sentence_embeddings")

                # 将每一个token的隐藏状态通过线性层，降维到300
                # sentence_embeddings_300 = linear(sentence_embeddings)

                # 将这句话的所有单词的词向量添加到word_embeddings列表中
                word_embeddings.append(sentence_embeddings)
            word_embeddings_tensor = torch.cat(word_embeddings, dim=0)
            # word_embeddings_list=word_embeddings_tensor.cpu().tolist()
            # with open("word_emb.txt", "a+") as file:
            # # Write the variable to the file
            #   file.write(str(word_embeddings_list))
            #   file.write("iter_")



            # print(word_embeddings_tensor.shape, "word_emb")
        else:

            sentence = label_dict

            marked_sentence = sentence

            # 使用tokenizer将tokens转为input_ids
            input_ids = tokenizer.convert_tokens_to_ids(marked_sentence)

            # 将input_ids转为Tensor，并添加一个新的维度以符合BERT的输入需求
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)

            # 获取BERT模型的输出
            outputs = model(input_tensor)

            # 获取每一个token的隐藏状态
            sentence_embeddings = outputs[0]

            # 将每一个token的隐藏状态通过线性层，降维到300
            # sentence_embeddings_300 = linear(sentence_embeddings)

            # 将这句话的所有单词的词向量添加到word_embeddings列表中
            word_embeddings.append(sentence_embeddings)
            word_embeddings_tensor = torch.cat(word_embeddings, dim=0)[0]
            # with open("label_emb.txt", "w") as file:
            # # Write the variable to the file
            #   file.write(str(word_embeddings_tensor))

            # print(word_embeddings_tensor.shape, "word_emb")
        # [sentences_num, words_num, hidden_size]

        # embedding = self.embedding(vocab_id_list)
        print(word_embeddings_tensor.shape,"word_embeddings_tensor")
        return (word_embeddings_tensor)
