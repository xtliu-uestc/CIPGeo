import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, scatter, softmax
from torch_geometric.nn.norm import GraphNorm, BatchNorm

class LightGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=False, bias=False):
        super(LightGAT, self).__init__(aggr='add')
        self.add_self_loops = add_self_loops
        self.lin_src = nn.Linear(in_channels, out_channels)
        self.lin_dst = nn.Linear(in_channels, out_channels)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x, edge_index, edge_att=None):  
        lat_lon = x[:, -2:] # Using the last two geographical coordinates for propagation
        x_src = self.lin_src(x)   
        x_dst = self.lin_dst(x)   
        h = (x_src, x_dst)

        alpha = self.edge_updater(edge_index, h=h)
        feat = torch.cat([x_src, lat_lon], dim=-1)
        out = self.propagate(edge_index, x=lat_lon, alpha=alpha, edge_att=edge_att)

        if self.bias is not None:
            out = out + self.bias

        return out
    
    def edge_update(self, h_j, h_i, index):
        temp = h_j.shape[1] ** 0.5
        alpha = torch.sum((h_j / temp) * h_i, dim=-1)
        alpha = softmax(alpha, index)
        return alpha

    def message(self, x_j, alpha, edge_att):    
        m = alpha.unsqueeze(-1) * x_j
        if edge_att is not None:
            m = m * edge_att
        return m

class QRMLPGeo(nn.Module):
    def __init__(self, dim_in, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pred = nn.Linear(hidden, 4) 

    def forward(self, x, edge_index, tg_mask):
        z = self.mlp(x)
        pred = self.pred(z[tg_mask == 1])
        return pred


class GAT_PointGeo(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.conv = LightGAT(dim_in, dim_in)
        self.pred = nn.Linear(dim_in, 2)

    def forward(self, x, edge_index, tg_mask):
        z = self.conv(x, edge_index)
        pred = self.pred(z[tg_mask==1])
        return pred


class QRGATGeo(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.conv = LightGAT(dim_in, dim_in)
        self.pred = nn.Linear(2, 4)
    
    def forward(self, x, edge_index, tg_mask):
        z = self.conv(x, edge_index)
        pred = self.pred(z[tg_mask==1])
        return pred