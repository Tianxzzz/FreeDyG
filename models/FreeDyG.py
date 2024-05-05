import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class FreeDyG(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, num_layers: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 128, device: str = 'cpu'):
        """
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(FreeDyG, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim,parameter_requires_grad=False)
        
        self.nif_feat_dim = self.channel_embedding_dim

        self.nif_encoder = NIFEncoder(nif_feat_dim=self.nif_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.edge_feat_dim, bias=True),
            'nif': nn.Linear(in_features=self.nif_feat_dim, out_features=self.edge_feat_dim, bias=True)
        })
        self.reduce_layer = nn.Linear(4*self.edge_feat_dim, self.edge_feat_dim)
       
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=max_input_sequence_length, num_channels=self.edge_feat_dim,
                     token_dim_expansion_factor=0.5,
                     channel_dim_expansion_factor=4, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        # self.transformers = nn.ModuleList([
        #     TransformerEncoder(seq_len=max_input_sequence_length,attention_dim=self.edge_feat_dim, num_heads=self.num_heads, dropout=self.dropout)
        #     for _ in range(self.num_layers)
        # ])
       
        self.weightagg = nn.Linear(self.edge_feat_dim,1)
       

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        src_nodes_neighbor_ids, src_nodes_edge_ids, src_nodes_neighbor_times = \
           self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

       
        dst_nodes_neighbor_ids, dst_nodes_edge_ids, dst_nodes_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.max_input_sequence_length)

        # src_nodes_nif_features, Tensor, shape (batch_size, src_max_seq_length, nif_feat_dim)
        # dst_nodes_nif_features, Tensor, shape (batch_size, dst_max_seq_length, nif_feat_dim)
        src_nodes_nif_features, dst_nodes_nif_features = \
            self.nif_encoder(src_node_ids=src_node_ids,dst_node_ids=dst_node_ids,src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                                                dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

        # get the features of the sequence of source and destination nodes
        # src_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_nodes_neighbor_ids,
                              nodes_edge_ids=src_nodes_edge_ids, nodes_neighbor_times=src_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=dst_nodes_neighbor_ids,
                              nodes_edge_ids=dst_nodes_edge_ids, nodes_neighbor_times=dst_nodes_neighbor_times, time_encoder=self.time_encoder)
             # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_edge_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * edge_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        
        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)
        src_nodes_nif_features = self.projection_layer['nif'](src_nodes_nif_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_nodes_neighbor_node_raw_features)
        dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        dst_nodes_neighbor_time_features = self.projection_layer['time'](dst_nodes_neighbor_time_features)
        dst_nodes_nif_features = self.projection_layer['nif'](dst_nodes_nif_features)

        src_combined_features = torch.cat([src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features,src_nodes_nif_features], dim=-1)
        dst_combined_features = torch.cat([dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features,dst_nodes_nif_features], dim=-1)
      
        src_combined_features = self.reduce_layer(src_combined_features)
        dst_combined_features = self.reduce_layer(dst_combined_features)
       
        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            src_combined_features = mlp_mixer(input_tensor=src_combined_features)
        for mlp_mixer in self.mlp_mixers:
            # Tensor, shape (batch_size, num_neighbors, num_channels)
            dst_combined_features = mlp_mixer(input_tensor=dst_combined_features)
        
        # for transformer in self.transformers:
        #     src_combined_features = transformer(src_combined_features)
        # for transformer in self.transformers:
        #     dst_combined_features = transformer(dst_combined_features)
    

        src_weight = self.weightagg(src_combined_features).transpose(1, 2)
        dst_weight = self.weightagg(dst_combined_features).transpose(1, 2)
       
        src_combined_features = src_weight.matmul(src_combined_features).squeeze(dim=1)
        dst_combined_features = dst_weight.matmul(dst_combined_features).squeeze(dim=1)
      
       
        return src_combined_features, dst_combined_features


    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        

        # ndarray, set the time features to all zeros for the padded timestamp
        nodes_neighbor_time_features[torch.from_numpy(nodes_neighbor_ids == 0)] = 0.0
    

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features


    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

class NIFEncoder(nn.Module):

    def __init__(self, nif_feat_dim: int, device: str = 'cpu'):
       
        super(NIFEncoder, self).__init__()

        self.nif_feat_dim = nif_feat_dim
        self.device = device

        self.nif_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.nif_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.nif_feat_dim, out_features=self.nif_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
       
        # two lists to store the appearances of source and destination nodes
        src_nodes_appearances, dst_nodes_appearances = [], []
        # src_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for i in range(len(src_node_ids)):
            src_node_id = src_node_ids[i]
            dst_node_id = dst_node_ids[i]
            src_node_neighbor_ids = src_nodes_neighbor_ids[i]
            dst_node_neighbor_ids = dst_nodes_neighbor_ids[i]

            # Calculate unique keys and counts for source and destination
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_node_neighbor_ids, return_inverse=True, return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_node_neighbor_ids, return_inverse=True, return_counts=True)

            # Create mappings from node IDs to their counts
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # Adjust counts specifically for the cases where src_node_id appears in dst's neighbors and vice versa
            if src_node_id in dst_mapping_dict:
                src_count_in_dst = dst_mapping_dict[src_node_id]
                src_mapping_dict[src_node_id] = src_count_in_dst
                dst_mapping_dict[src_node_id] = src_count_in_dst
            if dst_node_id in src_mapping_dict:
                dst_count_in_src = src_mapping_dict[dst_node_id]
                src_mapping_dict[dst_node_id] = dst_count_in_src
                dst_mapping_dict[dst_node_id] = dst_count_in_src

        # Calculate appearances in each other's lists
            src_node_neighbor_counts_in_dst = torch.tensor([dst_mapping_dict.get(neighbor_id, 0) for neighbor_id in src_node_neighbor_ids]).float().to(self.device)
            dst_node_neighbor_counts_in_src = torch.tensor([src_mapping_dict.get(neighbor_id, 0) for neighbor_id in dst_node_neighbor_ids]).float().to(self.device)

            # Stack counts to get a two-column tensor for each node list
            src_nodes_appearances.append(torch.stack([torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device), src_node_neighbor_counts_in_dst], dim=1))
            dst_nodes_appearances.append(torch.stack([dst_node_neighbor_counts_in_src, torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)], dim=1))

    # Stack to form batch tensors
        src_nodes_appearances = torch.stack(src_nodes_appearances, dim=0)
        dst_nodes_appearances = torch.stack(dst_nodes_appearances, dim=0)

        return src_nodes_appearances, dst_nodes_appearances


    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_nodes_neighbor_ids and dst_nodes_neighbor_ids
        :param src_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_nodes_appearances, dst_nodes_appearances = self.count_nodes_appearances(src_node_ids=src_node_ids,dst_node_ids=dst_node_ids,src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                                                                                                  dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, nif_feat_dim)
        
        # Tensor, shape (batch_size, dst_max_seq_length, nif_feat_dim)

        src_nodes_nif_features =  (src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features =  (dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)


        src_nodes_nif_features = self.nif_encode_layer(src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = self.nif_encode_layer(dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        
        # src_nodes_nif_features, Tensor, shape (batch_size, src_max_seq_length, nif_feat_dim)
        # dst_nodes_nif_features, Tensor, shape (batch_size, dst_max_seq_length, nif_feat_dim)
        return src_nodes_nif_features, dst_nodes_nif_features


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)

class FilterLayer(nn.Module):
    def __init__(self,max_input_length:int, hidden_dim: int):
        super(FilterLayer, self).__init__()

        self.max_input_length=max_input_length

        self.complex_weight = nn.Parameter(torch.randn(1,self.max_input_length//2+1, hidden_dim, 2,dtype=torch.float32))

        self.Dropout = nn.Dropout(0.1)

        self.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, input_tensor: torch.Tensor):
      

        batch,seq_len, hidden = input_tensor.shape

        hidden_states=input_tensor
      
       
        x = torch.fft.rfft(hidden_states, n=self.max_input_length, dim=1, norm='forward')
        weight = torch.view_as_complex(self.complex_weight)
        x=x*weight
        sequence_emb_fft = torch.fft.irfft(x, n=self.max_input_length, dim=1, norm='forward')

        sequence_emb_fft = sequence_emb_fft[:,0:seq_len,:]
        hidden_states = self.Dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)
        self.filter=FilterLayer(max_input_length=num_tokens,hidden_dim=num_channels)
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels, dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)
       
    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        hidden_tensor=self.filter(input_tensor)
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(hidden_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor


