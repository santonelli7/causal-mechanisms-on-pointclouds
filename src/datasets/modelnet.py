import os
import glob
import random
from typing import List, Optional
from pathlib import Path

import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.io import read_off

from src.datasets.data import TransformedSample

class TransformedModelNet(ModelNet):
    def __init__(self, root: Path, train: Optional[bool] = True, transform = None, pre_transform = None, transforms: Optional[List] = []):
        """
        ModelNet dataset where each data has a random transformation.

        :param root: Path to the folder that contains the dataset 
        :param train: download train dataset or test dataset
        :param transform: A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access
        :param pre_transform: A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk
        :param transforms: A list of functions/transforms. The data object will be transformed before being saved to disk
        """
        self.idx_to_transf = {idx: transf for idx, transf in enumerate(transforms)} 
        self.transf_to_idx = {transf: idx for idx, transf in self.idx_to_transf.items()}
        super(TransformedModelNet, self).__init__(root, name='10', train=train, transform=transform, pre_transform=pre_transform)

    def download(self):
        super(TransformedModelNet, self).download()

    def process(self):
        super(TransformedModelNet, self).process()

    def process_set(self, dataset):
        categories = glob.glob(os.path.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        # canonical distribution
        data_list = []
        # transformed distribution for each mechanism
        Q_n = {idx: [] for idx, _ in self.idx_to_transf.items()} 
        for target, category in enumerate(categories):
            folder = os.path.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))
            if category == "bed":
                for path in paths:
                    data = read_off(path)
                    data.y = torch.tensor([target])
                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            new_data_list = []
            for data in data_list:
                data = self.pre_transform(data)
                new_data_list.append(data)
                if len(self.idx_to_transf) != 0:
                    for idx, transf in self.idx_to_transf.items():
                        Q_n[idx].append(transf(data.clone()))
            data_list = new_data_list

        if len(self.idx_to_transf) != 0:
            new_data_list = []
            for data in data_list:
                # pick a random transform
                t = random.sample(list(self.idx_to_transf.keys()), 1)[0]
                # define the new Data with its transformed version
                t_data = TransformedSample(x=data.x, pos=data.pos, edge_index=data.edge_index, edge_attr=data.edge_attr, face=data.face, y=data.y, transf=t)
                # sample iid from the transformed distribution obtained by t
                sample = random.sample(Q_n[t], 1)[0]
                t_data.pos_transf = sample.pos
                new_data_list.append(t_data)
            data_list = new_data_list

        return self.collate(data_list)
