from torch import nn
import torch

from model.STN import SpatialTransformer
from model.TTN import TemporalTransformer
from model.embed import Embedding
from model.pred import Pred


class STBlock(nn.Module):
    def __init__(self, d_k, d_v, d_model, len_his, n_heads, n_nodes, dropout, adj):
        super(STBlock, self).__init__()
        self.STN = SpatialTransformer(d_k, d_v, d_model, len_his, n_heads, n_nodes, dropout, adj)
        self.TTN = TemporalTransformer(d_k, d_v, d_model, len_his, n_heads, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.norm1(self.STN(x) + x))
        x = self.dropout(self.norm2(self.TTN(x) + x))
        return x


class STTN_feature(nn.Module):
    def __init__(self, d_k=32, d_v=32, d_feature=1, d_model=32, len_his=3, len_pred=1, n_heads=8, n_nodes=1542,
                 n_layers=5, dropout=0.2, num_other_feature=10 ,adj=None):
        super(STTN_feature, self).__init__()

        self.embed = Embedding(d_feature=d_feature, d_model=d_model)
        self.st_blocks = nn.ModuleList(
            [
                STBlock(d_k, d_v, d_model, len_his, n_heads, n_nodes, dropout, adj) for _ in range(n_layers)
            ]
        )
        self.pred = Pred(d_feature=d_feature, d_model=d_model, len_his=len_his, len_pred=len_pred)
        self.linear = nn.Linear(num_other_feature+1,1)

    def forward(self, input):
        x = input[:,:,:,0]
        x = x.unsqueeze(-1)
        features = input[:,-1,:,1:]
        features = features.unsqueeze(1)

        x = self.embed(x)
        for layer in self.st_blocks:
            x = layer(x)
        x = self.pred(x)

        x = torch.cat([x,features],dim=-1)
        x = self.linear(x)

        return x
