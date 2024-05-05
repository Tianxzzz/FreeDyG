import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.load_configs import get_link_prediction_args
import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from scipy import stats
logging.getLogger('matplotlib').setLevel(logging.WARNING)



class DishTS(nn.Module):
    def __init__(self, dish_init, n_series, seq_len):
        super().__init__()
        init = dish_init #'standard', 'avg' or 'uniform'
        activate = True
        self.n_series = n_series # number of series
        self.lookback = seq_len # lookback length
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(self.n_series, self.lookback, 2)/self.lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(self.n_series, self.lookback, 2)/self.lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(self.n_series, self.lookback, 2)/self.lookback+torch.rand(self.n_series, self.lookback, 2)/self.lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(self.n_series)), nn.Parameter(torch.zeros(self.n_series))
        self.activate = activate

    def forward(self, batch_x, mode='forward', dec_inp=None):
        if mode == 'forward':
         
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x, dec_inp
        elif mode == 'inverse':
            
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        
        x_transpose = batch_x.permute(2,0,1) 
      
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:] 
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)

    def forward_process(self, batch_input):
        #print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst

    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih

class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)
        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))
        return output

class LayerNorm(nn.Module):
    def __init__(self, hidden_dim: int , eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class ProjectionLayer(nn.Module):
    def __init__(self, hidden_dim: int):

        super().__init__()
        self.fc1 = nn.Linear(2*hidden_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.act = nn.ReLU()

    def forward(self, input: torch.Tensor):
        h = self.act(self.fc1(input))
    
        h = self.fc2(h)
        
        return h
    
class onehotProjectionLayer(nn.Module):
    def __init__(self, onehot_dim:int,hidden_dim: int):

        super().__init__()
        self.onehot_dim=onehot_dim
        self.hidden_dim=hidden_dim
        # self.fc1 = nn.Linear(2*(hidden_dim + onehot_dim), hidden_dim)
        self.src_fc = nn.Linear(self.hidden_dim + self.onehot_dim, self.hidden_dim)
        self.dst_fc = nn.Linear(self.hidden_dim + self.onehot_dim, self.hidden_dim)
        # self.fc1 = nn.Linear(2*hidden_dim, 1)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.act = nn.ReLU()

    def forward(self, input: torch.Tensor):
        h_src = self.src_fc(input[:,:self.hidden_dim + self.onehot_dim])
        h_dst = self.dst_fc(input[:,self.hidden_dim + self.onehot_dim:2 * (self.hidden_dim + self.onehot_dim)])
        h = self.fc2(self.act(h_src+h_dst))
        # h = self.fc1(input)
        return h

class NewLayer(nn.Module):
    def __init__(self, hidden_dim:int):
        super(NewLayer, self).__init__()
        
       
        self.outputlayer = ProjectionLayer(hidden_dim)
      


    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        hidden_states = torch.cat([input_1, input_2], dim=1)
        
        output = self.outputlayer(hidden_states)
    
        return output

class onehotLayer(nn.Module):
    def __init__(self, onehot_dim:int,hidden_dim:int, filter:str):
        super(onehotLayer, self).__init__()
        self.filters=filter
        
        self.outputlayer = onehotProjectionLayer(onehot_dim,hidden_dim)
        


    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        hidden_states = torch.cat([input_1, input_2], dim=1)

        output = self.outputlayer(hidden_states)

        return output


class MergeLayer(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
#         self.aug = DistributionUncertainty()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)

#         x = self.aug(x)
        # # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h



class HiddenLayer(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int):

        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.act = nn.ReLU()
        # self.aug = DistributionUncertainty()
    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):

        # Tensor, shape (*, input_dim1 + input_dim2)
        x = torch.cat([input_1, input_2], dim=1)
        # x=input_1+input_2
        x= self.act(self.fc1(x))
        # x_aug=self.aug(x)
        return x

class DownstreamLayer(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output

        """
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, output_dim)
        # self.act = nn.ReLU()
        # self.aug = DistributionUncertainty()
        # self.dropout = nn.Dropout(0.1)
    def forward(self, input: torch.Tensor, stat:str):

        # x = self.aug(x)
        h = self.fc1(input)
        # h = self.fc3(self.act(self.fc1(x)))
#         h = self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))
        # Tensor, shape (*, output_dim)
        # h = self.fc3(self.act(self.fc2(self.aug(self.act(self.fc1(x))))))
        return h


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, filter: str, dropout: float = 0.1):
        """
        Multi-Layer Perceptron Classifier.
        :param input_dim: int, dimension of input
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.filters=filter
        # if self.filters=='adap':
        self.input_dim=input_dim

        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        multi-layer perceptron classifier forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        # Tensor, shape (*, 80)
        x = self.dropout(self.act(self.fc1(x)))
        # Tensor, shape (*, 10)
        x = self.dropout(self.act(self.fc2(x)))
        # Tensor, shape (*, 1)
        return self.fc3(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        """
        Multi-head Attention module.
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim: int, dimension of time features (time encodings)
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(MultiHeadAttention, self).__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim

        assert self.query_dim % num_heads == 0, "The sum of node_feat_dim and time_feat_dim should be divided by num_heads!"

        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)

        self.scaling_factor = self.head_dim ** -0.5

        self.layer_norm = nn.LayerNorm(self.query_dim)
       
        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)

        self.dropout = nn.Dropout(dropout)
        # self.complex_weight = nn.Parameter(torch.randn(10//2+1,node_feat_dim + edge_feat_dim + time_feat_dim , 2,dtype=torch.float32))
        #  self.LayerNorm = LayerNorm(node_feat_dim + edge_feat_dim + time_feat_dim, eps=1e-12)

    def forward(self, node_features: torch.Tensor, node_time_features: torch.Tensor, neighbor_node_features: torch.Tensor,
                neighbor_node_time_features: torch.Tensor, neighbor_node_edge_features: torch.Tensor, neighbor_masks: np.ndarray):
        """
        temporal attention forward process
        :param node_features: Tensor, shape (batch_size, node_feat_dim)
        :param node_time_features: Tensor, shape (batch_size, 1, time_feat_dim)
        :param neighbor_node_features: Tensor, shape (batch_size, num_neighbors, node_feat_dim)
        :param neighbor_node_time_features: Tensor, shape (batch_size, num_neighbors, time_feat_dim)
        :param neighbor_node_edge_features: Tensor, shape (batch_size, num_neighbors, edge_feat_dim)
        :param neighbor_masks: ndarray, shape (batch_size, num_neighbors), used to create mask of neighbors for nodes in the batch
        :return:
        """
        # Tensor, shape (batch_size, 1, node_feat_dim)
        node_features = torch.unsqueeze(node_features, dim=1)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        
        # shape (batch_size, 1, num_heads, self.head_dim)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_neighbors, node_feat_dim + edge_feat_dim + time_feat_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)

        
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        # Tensor, shape (batch_size, num_neighbors, num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        # Tensor, shape (batch_size, num_heads, num_neighbors, self.head_dim)
        value = value.permute(0, 2, 1, 3)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention = torch.einsum('bhld,bhnd->bhln', query, key)
        attention = attention * self.scaling_factor

        # Tensor, shape (batch_size, 1, num_neighbors)
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1)
        attention_mask = attention_mask == 0
        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)

        # Tensor, shape (batch_size, self.num_heads, 1, num_neighbors)
        # note that if a node has no valid neighbor (whose neighbor_masks are all zero), directly set the masks to -np.inf will make the
        # attention scores after softmax be nan. Therefore, we choose a very large negative number (-1e10 following TGAT) instead of -np.inf to tackle this case
        attention = attention.masked_fill(attention_mask, -1e10)

        # Tensor, shape (batch_size, num_heads, 1, num_neighbors)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))

        # Tensor, shape (batch_size, num_heads, 1, self.head_dim)
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)

        # Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.dropout(self.residual_fc(attention_output))

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        output = self.layer_norm(output + residual)

        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        output = output.squeeze(dim=1)
        # Tensor, shape (batch_size, num_heads, num_neighbors)
        attention_scores = attention_scores.squeeze(dim=2)

        return output, attention_scores


class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0, 1), inputs_key.transpose(0, 1), inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs
