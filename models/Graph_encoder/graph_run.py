import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import BertModel
from graph import GraphEncoder
import os


batch_size = 2
sequence_length = 256
hidden_size = 768
label_num = 103

inputs_embeds = torch.randn(batch_size, sequence_length, hidden_size)
attention_mask = torch.ones(batch_size, sequence_length)
labels = torch.randint(0, 2, (batch_size, label_num))#one hot matrix of labels
print(labels)
pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
config=pretrained_bert.config
print(config.hidden_size)

model = GraphEncoder(config=pretrained_bert.config, graph=False, layer=1, data_path="../data/rcv1", threshold=0.01, tau=1)


def embeddings(label_name):
    return torch.randn(label_name.size(1), label_name.size(0))
print(inputs_embeds.shape,"inputs_embeds")
print(attention_mask.shape,"attention_mask")
print(labels.shape,"labels")
print(embeddings,"embeddings")

model_bert = BertModel(config)
output = model.forward(inputs_embeds, attention_mask, labels, lambda x: model_bert.embeddings(x)[0])


print(output)
print(output.shape)
