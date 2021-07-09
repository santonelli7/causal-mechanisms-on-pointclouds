from src.arguments import Args
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, Dropout
import torch.nn.functional as F

from torch_geometric.nn import DynamicEdgeConv, global_max_pool, BatchNorm as BN

def MLP(channels):
    return Seq(*[
                Seq(
                    Lin(channels[i - 1], channels[i]), BN(channels[i], momentum=0.9), LeakyReLU(negative_slope=0.2)
                ) for i in range(1, len(channels))
            ])

# G_FEAT = [96, 256, 256, 256, 128, 128, 128, 3]
class Expert(nn.Module):
    def __init__(self, args: Args):
        super(Expert, self).__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k=args.k, aggr=args.aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k=args.k, aggr=args.aggr)
        self.lin = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, args.in_channels))
        

    def forward(self, pos: torch.Tensor, batch:torch.tensor) -> torch.Tensor:
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin(torch.cat([x1, x2], dim=-1))
        out = self.mlp(out)
        return torch.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, args: Args):
        channels = args.D_channels
        self.layer_num = len(channels)-1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(channels[inx], channels[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(channels[-1], channels[-1]),
                                         nn.Linear(channels[-1], channels[-2]),
                                         nn.Linear(channels[-2], channels[-2]),
                                         nn.Linear(channels[-2], 1))

    def forward(self, pos):
        """
            pos: B x N x 3
        """
        feat = pos.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out) # (B, 1)

        return torch.sigmoid(out)
