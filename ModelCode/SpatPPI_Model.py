import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os
from torch.nn.utils.weight_norm import weight_norm


class DataToGraph(data.Dataset):
    def __init__(self, list_IDs, df, Tensor_folder):
        self.list_IDs = list_IDs
        self.df = df
        self.folder = Tensor_folder

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        
        p1 = self.df.iloc[index]['p1']
        p1_X = torch.load(self.folder+f'{p1}_X.tensor')
        p1_O = torch.load(self.folder+f'{p1}_O.tensor')
        if os.path.exists(self.folder+f'{p1}_new__node_feature.tensor'):
            p1_V = torch.load(self.folder+f'{p1}_new__node_feature.tensor')
        else:
            p1_V = torch.load(self.folder+f'{p1}_node_feature.tensor')
        p1_V = torch.load(self.folder+f'{p1}_node_feature.tensor')
        p1_mask = torch.load(self.folder+f'{p1}_mask.tensor')

        p2 = self.df.iloc[index]['p2']
        p2_X = torch.load(self.folder+f'{p2}_X.tensor')
        p2_O = torch.load(self.folder+f'{p2}_O.tensor')
        if os.path.exists(self.folder+f'{p2}_new__node_feature.tensor'):
            p2_V = torch.load(self.folder+f'{p2}_new__node_feature.tensor')
        else:
            p2_V = torch.load(self.folder+f'{p2}_node_feature.tensor')
        p2_mask = torch.load(self.folder+f'{p2}_mask.tensor')
        
        Protein_1 = {'PDB_NAME': p1, 'X': p1_X, 'O': p1_O, 'V': p1_V, 'mask': p1_mask}
        Protein_2 = {'PDB_NAME': p2, 'X': p2_X, 'O': p2_O, 'V': p2_V, 'mask': p2_mask}
        y = self.df.iloc[index]['label']

        return Protein_1, Protein_2, y

class Normalize(nn.Module): 
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias

def gather_coordinate(coordinates, neighbor_idx):
    
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, coordinates.size(2))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, coordinates.size(3))
    neighbor_features = torch.gather(coordinates, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [3] + [3])
    
    return neighbor_features

def gather_nodes(nodes, neighbor_idx):

    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn, h_nodes

class SpatPPI(nn.Module):
    def __init__(self, **config):
        super(SpatPPI, self).__init__()
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        protein_node_dim = config["PROTEIN"]["NODE_DIM"]
        protein_edge_dim = config["PROTEIN"]["EDGE_DIM"]
        attention_heads = config["ATTENTION"]["HEADS"]
        attention_beta = config["ATTENTION"]["HEADS"]
        attention_neighbors = config["ATTENTION"]["NEIGHBORS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]
        bcc_heads = config["BCC"]["HEADS"]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Graph_embedding = PEMA(protein_node_dim, protein_edge_dim, protein_emb_dim,
                                    attention_heads, attention_beta, attention_neighbors, device)

        self.bcn = weight_norm(BCCLayer(hidden_dim = protein_emb_dim, device = device),name='h_mat', dim=None)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def attention_pooling(self, fusion_logits):
        fusion_logits = self.p_net(fusion_logits.unsqueeze(1)).squeeze(1)
        return fusion_logits

    def forward(self, V1, E1, E_idx1, mask1, V2, E2, E_idx2, mask2, mode="train"):
        emb1 = self.Graph_embedding(V1, E1, E_idx1, mask1) 
        emb2 = self.Graph_embedding(V2, E2, E_idx2, mask2)
        f = self.bcn(emb1, emb2, mask1, mask2)
        score = self.mlp_classifier(f)
        if mode == "train":
            return emb1, emb2, f, score
        elif mode == "eval":
            return emb1, emb2, score, f


class PEMA(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, nheads, beta, k_neighbors, device, augment_eps=0., dropout=0.2):
        super(PEMA, self).__init__()
        #hyperparameterization
        self.augment_eps = augment_eps
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim[0]
        self.device = device
        #coding layer
        self.W_s =  nn.Linear(node_features, hidden_dim[0], bias=True)
        self.W_t =  nn.Linear(edge_features, hidden_dim[0], bias=True)
        #encoders
        self.encoder_layers = nn.ModuleList([
            EAttentionLayer(hidden_dim[i], hidden_dim[i]*2, nheads, device, dropout=dropout)
            for i in range(len(hidden_dim))
        ])
        self.norm = Normalize(hidden_dim[0])
        self.fc1 = nn.Linear(hidden_dim[0]*3, hidden_dim[0])
        #weight Initialization
        for p in self.parameters():
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p)
        
    def forward(self, V, E, E_idx, mask):
        """V.shape(B, L, node_features)
           E.shape(B, L, K, edge_features)
           E.idx.shape(B, L, K)
           h_E.shape(B, L, K, hidden_dim)
           h_V.shape(B, L, hidden_dim)
           h_KV.shape(B, L, K, hidden_dim)
           h_EV.shape(B, L, K, hidden_dim*2)
           mask_attend(B, L, K)
        """
        h_V = self.W_s(V)
        h_E = self.W_t(E)
        B, L, K = h_E.shape[:3]
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend 
        for layer in self.encoder_layers:
            h_EV, h_KV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, h_KV, h_E, mask_V=mask, mask_attend=mask_attend)
            New_h_E, _ = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_d = h_V.unsqueeze(2).expand(B, L, K, self.hidden_dim)
            h_E = F.relu(self.fc1(torch.cat([New_h_E, h_V_d], -1)))
            h_E = self.norm(h_E)
        return h_V
    
class EAttentionLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads, device, dropout=0.2):
        super(EAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.attention = NeighborAttention(num_hidden, num_in, num_heads, device)
        
    def forward(self, h_V, h_EV, h_KV, h_KE, mask_V=None, mask_attend=None): 
        

        dh = self.attention(h_V, h_EV, h_KV, h_KE, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))
        


        if mask_V is not None: 
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        
        return h_V



class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads, device):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.device = device

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K1 = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K2 = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, device, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(device))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def QKK(self, Q, K1, K2):
        result_multiply = torch.mul(Q, K1)
        result_multiply = torch.mul(result_multiply, K2)
        result = torch.sum(result_multiply, dim=-1)
        return result
    
    def forward(self, h_V, h_EV, h_KV, h_KE, mask_attend=None):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_EV:           Cat Node-Edge features  [N_batch, N_nodes, K, N_hidden*2]
            h_KV:           Neighbor features       [N_batch, N_nodes, K, N_hidden]
            h_KE:           Edge features           [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]           
        Returns:
            h_V:            Node update
        """
        n_batch, n_nodes, n_neighbors = h_EV.shape[:3]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, d])
        Q = Q.repeat(1, 1, n_neighbors, 1, 1)
        K1 = self.W_K1(h_KE).view([n_batch, n_nodes, n_neighbors, n_heads, d]) 
        K2 = self.W_K2(h_KV).view([n_batch, n_nodes, n_neighbors, n_heads, d])
        V = self.W_V(h_EV).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = self.QKK(Q,K1,K2).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / d
        
        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(2).expand(-1,-1,n_heads,-1)
            attend = self._masked_softmax(attend_logits, mask, self.device) # [B, L, heads, K]
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2,3)) # [B, L, heads, 1, K] × [B, L, heads, K, d]
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update
    
class BCCLayer(nn.Module):
    def __init__(self, hidden_dim, device, act='ReLU', dropout=0.2, k=2):
        super(BCCLayer, self).__init__()

        self.k = k
        self.hidden_dim = hidden_dim[0]
        self.Q_net = FCNet([self.hidden_dim, self.hidden_dim * self.k], act=act, dropout=dropout)
        self.K_net = FCNet([self.hidden_dim, self.hidden_dim * self.k], act=act, dropout=dropout)
        self.h_mat = nn.Parameter(torch.Tensor(1, 1, self.hidden_dim * self.k).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, 1, 1).normal_())
        self.p_net = nn.AvgPool2d(kernel_size=(2000, 1))
        self.bn = nn.BatchNorm1d(self.hidden_dim * self.k)
        self.device = device

    def attention_pooling(self, fusion_logits):
        fusion_logits = self.p_net(fusion_logits.unsqueeze(1)).squeeze(1)
        return fusion_logits

    def _masked_softmax(self, att_maps, mask_attend, dim, device):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, att_maps, torch.tensor(negative_inf).to(device))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, X, Y, mask1, mask2):

        A1_X = self.Q_net(X)
        A1_Y = self.K_net(Y)
        A2_Y = self.Q_net(Y)
        A2_X = self.K_net(X)
        A1 = torch.einsum('xyk,bvk,bqk->bvq', (self.h_mat, A1_X, A1_Y)) + self.h_bias
        A2 = torch.einsum('xyk,bqk,bvk->bqv', (self.h_mat, A2_Y, A2_X)) + self.h_bias
        mask1_attend = mask1.unsqueeze(dim=-2)
        mask1_attend = mask1_attend.expand(A1_X.shape[0], 2000, 2000)
        mask2_attend = mask2.unsqueeze(dim=-2)
        mask2_attend = mask2_attend.expand(A2_Y.shape[0], 2000, 2000)

        alpha = self._masked_softmax(A1, mask2_attend, 1, self.device)
        logits_X = torch.einsum('bvq, bqk->bvk', (alpha, A1_Y))
        beta = self._masked_softmax(A2, mask1_attend, 1, self.device)
        logits_Y = torch.einsum('bqv, bvk->bqk', (beta, A2_X))
        logits_X_mask = mask1.unsqueeze(dim=-1)
        logits_X_mask = logits_X_mask.expand(logits_X.shape[0], 2000, logits_X.shape[2])
        logits_X = logits_X * logits_X_mask
        logits_Y_mask = mask2.unsqueeze(dim=-1)
        logits_Y_mask = logits_Y_mask.expand(logits_Y.shape[0], 2000, logits_Y.shape[2])
        logits_Y = logits_Y * logits_Y_mask
        logits_plus = self.attention_pooling(logits_X).squeeze(1) + self.attention_pooling(logits_Y).squeeze(1)
        logits = self.bn(logits_plus)
        return logits


class FCNet(nn.Module):

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x