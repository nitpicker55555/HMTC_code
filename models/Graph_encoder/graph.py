import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import os

from torch_geometric.nn import GCNConv, GATConv

# GRAPH = 'GCN'
GRAPH = "GRAPHORMER"


# GRAPH = 'GAT'


class SelfAttention(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.confg_input=config

        self.self = BartAttention(config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob).to('cuda')
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps).to('cuda')
        self.dropout = nn.Dropout(config.hidden_dropout_prob).to('cuda').to('cuda')

    def forward(self, hidden_states,
                attention_mask=None, output_attentions=False, extra_attn=None):
        residual = hidden_states
        # print(hidden_states.shape, "SelfAttention input hidden_states")
        hidden_states, attn_weights, _ = self.self(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions,
            extra_attn=extra_attn,
        )

        hidden_states.to('cuda')
        # print(hidden_states.shape, "BartAttention output hidden_states")
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        outputs = (hidden_states,)
        # print(hidden_states.shape,"SelfAttention output hidden_states")

        if output_attentions:
            outputs += (attn_weights,)
        # print(outputs,"SelfAttention output")
        return outputs


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to('cuda')
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to('cuda')
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to('cuda')  #一个权重矩阵，768*768，所以输入的第一维度应该是768
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias).to('cuda')

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """Input shape: Batch x Time x Channel"""
        hidden_states.to('cuda')
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        # print(hidden_states.shape, "BartAttention input hidden_states")
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj

        # print(self.scaling, "self.scaling")
        query_states = self.q_proj(hidden_states).to('cuda') * self.scaling

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights += extra_attn

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)
        # print(attn_output.shape, attn_weights_reshaped.shape, past_key_value.shape,"BartAttention output_sum")
        return attn_output, attn_weights_reshaped, past_key_value


class GraphLayer(nn.Module):
    def __init__(self, config, last=False):
        super(GraphLayer, self).__init__()
        self.config = config

        class _Actfn(nn.Module):
            def __init__(self):
                super(_Actfn, self).__init__()
                if isinstance(config.hidden_act, str):
                    self.intermediate_act_fn = ACT2FN[config.hidden_act]
                else:
                    self.intermediate_act_fn = config.hidden_act

            def forward(self, x):
                return self.intermediate_act_fn(x)

        if GRAPH == 'GRAPHORMER':
            self.hir_attn = SelfAttention(config)
        elif GRAPH == 'GCN':
            self.hir_attn = GCNConv(config.hidden_size, config.hidden_size)
        elif GRAPH == 'GAT':
            self.hir_attn = GATConv(config.hidden_size, config.hidden_size, 1)

        self.last = last
        if last:
            #input config.hidden_size int 768
            self.cross_attn = BartAttention(config.hidden_size, 8, 0.1, True).to('cuda')
            self.cross_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps).to('cuda')
            self.classifier = nn.Linear(config.hidden_size, config.num_labels).to('cuda')
        self.output_layer = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size),
                                          _Actfn(),
                                          nn.Linear(config.intermediate_size, config.hidden_size),
                                          )
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, label_emb, extra_attn, self_attn_mask, inputs_embeds, cross_attn_mask):
        label_emb.to('cuda')
        self_attn_mask.to('cuda')
        inputs_embeds.to('cuda')
        cross_attn_mask.to('cuda')
        # print(label_emb.shape,"label_emb in GraphLayer & SelfAttention's hidden_states")
        if GRAPH == 'GRAPHORMER':
            label_emb = self.hir_attn(label_emb,
                                      attention_mask=self_attn_mask, extra_attn=extra_attn)[0]
            # label_emb = self.output_layer_norm(self.dropout(self.output_layer(label_emb)) + label_emb)
        elif GRAPH == 'GCN' or GRAPH == 'GAT':
            label_emb = self.hir_attn(label_emb.squeeze(0), edge_index=extra_attn)
        if self.last:
            label_emb = label_emb.expand(inputs_embeds.size(0), -1, -1)
            label_emb = self.cross_attn(inputs_embeds, label_emb,
                                        attention_mask=cross_attn_mask.unsqueeze(1), output_attentions=True,
                                        only_attn=True)
            return label_emb

        label_emb = self.output_layer_norm(self.dropout(self.output_layer(label_emb)) + label_emb)
        if self.last:
            label_emb = self.dropout(self.classifier(label_emb))
        return label_emb


class GraphEncoder(nn.Module):

    def __init__(self, config, graph=False, layer=1, data_path=None, threshold=0.01, tau=1):
        super(GraphEncoder, self).__init__()
        # print(config,"config Graph_encoder")

        self.config = config
        self.tau = tau
        self.label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}
        self.label_name = []
        # print(self.label_dict)
        #self.label_dict {0: 'ccat', 1: 'ecat', 2: 'gcat', 3: 'mcat', 4: 'c11', 5: 'c12', 6: 'c13', 7: 'c14', 8: 'c15', 9: 'c16', 10: 'c17', 11: 'c18', 12
        for i in range(len(self.label_dict)):
            self.label_name.append(self.label_dict[i])
        # print(self.label_name,"self.label_name original ")  #这时['ccat', 'ecat', 'gcat', 'mcat', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c1....
        self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
        # print(self.label_name,len(self.label_name),"self.label_name tokenizer")  # [[101, 10507, 4017, 102, 0], [101, 14925, 4017, 102, 0], [101, 1043, 11266, 102, 0],
        self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False).to('cuda')
        # print(self.label_name,self.label_name.shape, "self.label_name nn.Parameter") # transfer to tensor

        self.hir_layers = nn.ModuleList([GraphLayer(config, last=i == layer - 1) for i in range(layer)])

        self.label_num = len(self.label_name)

        self.graph = graph
        self.threshold = threshold

        if graph:
            label_hier = torch.load(os.path.join(data_path, 'slot.pt'))
            path_dict = {}
            num_class = 0
            for s in label_hier:
                print(s, "s")
                for v in label_hier[s]:
                    print(v, "v")
                    path_dict[v] = s
                    if num_class < v:
                        num_class = v
            if GRAPH == 'GRAPHORMER':
                num_class += 1
                for i in range(num_class):
                    if i not in path_dict:
                        path_dict[i] = i
                self.inverse_label_list = {}

                def get_root(path_dict, n):
                    ret = []
                    while path_dict[n] != n:
                        ret.append(n)
                        n = path_dict[n]
                    ret.append(n)
                    return ret

                for i in range(num_class):
                    self.inverse_label_list.update({i: get_root(path_dict, i) + [-1]})
                label_range = torch.arange(len(self.inverse_label_list))
                self.label_id = label_range
                node_list = {}

                def get_distance(node1, node2):
                    p = 0
                    q = 0
                    node_list[(node1, node2)] = a = []
                    node1 = self.inverse_label_list[node1]
                    node2 = self.inverse_label_list[node2]
                    while p < len(node1) and q < len(node2):
                        if node1[p] > node2[q]:
                            a.append(node1[p])
                            p += 1

                        elif node1[p] < node2[q]:
                            a.append(node2[q])
                            q += 1

                        else:
                            break
                    return p + q

                self.distance_mat = self.label_id.reshape(1, -1).repeat(self.label_id.size(0), 1)
                hier_mat_t = self.label_id.reshape(-1, 1).repeat(1, self.label_id.size(0))
                self.distance_mat.map_(hier_mat_t, get_distance)
                self.distance_mat = self.distance_mat.view(1, -1)
                self.edge_mat = torch.zeros(len(self.inverse_label_list), len(self.inverse_label_list), 15,
                                            dtype=torch.long)
                for i in range(len(self.inverse_label_list)):
                    for j in range(len(self.inverse_label_list)):
                        edge_list = node_list[(i, j)]
                        self.edge_mat[i, j, :len(edge_list)] = torch.tensor(edge_list) + 1
                self.edge_mat = self.edge_mat.view(-1, self.edge_mat.size(-1))

                self.id_embedding = nn.Embedding(len(self.inverse_label_list) + 1, config.hidden_size,
                                                 len(self.inverse_label_list))
                self.distance_embedding = nn.Embedding(20, 1, 0)
                self.edge_embedding = nn.Embedding(len(self.inverse_label_list) + 1, 1, 0)
                self.label_id = nn.Parameter(self.label_id, requires_grad=False)
                self.edge_mat = nn.Parameter(self.edge_mat, requires_grad=False)
                self.distance_mat = nn.Parameter(self.distance_mat, requires_grad=False)
            self.edge_list = [[v, i] for v, i in path_dict.items()]
            self.edge_list += [[i, v] for v, i in path_dict.items()]
            self.edge_list = nn.Parameter(torch.tensor(self.edge_list).transpose(0, 1), requires_grad=False)

    # outputs['inputs_embeds'],attention_mask, labels, lambda x: self.bert.embeddings(x)[0]
    def forward(self, label_emb):

        # print(inputs_embeds.shape,"inputs_embeds.shape")
        # print(attention_mask.shape,"attention_mask")
        # print(labels.shape,"labels")

        #label_mask = self.label_name != self.tokenizer.pad_token_id

        # print(self.label_name.shape, "label_name")
        # print(self.tokenizer.pad_token_id, " self.tokenizer.pad_token_id")
        # full name
        # torch.Size([103, 5]) label_name

        # label_emb = embeddings(self.label_name.to('cuda').to('cuda'))
        #
        # # torch.Size([5, 103]) label_emb
        #
        # # print(label_emb.shape,"label_emb GE")
        # # print(label_mask.shape)
        #
        # label_emb = (label_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)
        # label_emb = label_emb.unsqueeze(0)


        # print(label_emb.shape, "label_emb GE")
        # label_emb (1, label_num, hidden_size)

        expand_size = label_emb.size(-2) // self.label_name.size(0)
        if self.graph:
            if GRAPH == 'GRAPHORMER':
                label_emb += self.id_embedding(self.label_id[:, None].expand(-1, expand_size)).view(1, -1,
                                                                                                    self.config.hidden_size)
        print(label_emb.shape, "label_emb GE")
        print(len(self.hir_layers),"self.hir_layers")
        # for hir_layer in self.hir_layers:
        #     label_emb = hir_layer(label_emb, extra_attn, self_attn_mask, inputs_embeds, cross_attn_mask)
        # print(label_emb.shape,"label_emb")
        # label_emb (1, label_num, hidden_size)
        # token_probs = label_emb.mean(dim=1).view(attention_mask.size(0), attention_mask.size(1),
        #                                          self.label_name.size(0),
        #                                          )
        # # token_probs (batch_size, sequence_length, label_num) average value in 1 dim of label_emb
        # # print(token_probs.shape,"token_probs")
        # # print(token_probs)
        # # contrast_mask = (F.gumbel_softmax(token_probs, hard=False, dim=-1, tau=self.tau) * labels.unsqueeze(1)).sum(
        #     # -1)
        # #contrast_mask = (F.gumbel_softmax(token_probs, hard=False, dim=-1, tau=self.tau) * labels).sum(dim=1)
        # contrast_mask = (F.gumbel_softmax(token_probs, hard=False, dim=-1, tau=self.tau) * labels.unsqueeze(1)).sum(
        #      -1)
        # # contrast_mask= gumbel_softmax TO token_probs
        # temp = self.threshold
        # _mask = contrast_mask > temp
        # contrast_mask = contrast_mask + (1 - contrast_mask).detach()
        #
        # contrast_mask = contrast_mask * _mask
        # # print(contrast_mask.shape, "contrastive_mask")
        # contrastive_mask (batch_size, sequence_length, label_num)
        return label_emb
