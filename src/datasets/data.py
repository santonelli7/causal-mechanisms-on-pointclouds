from typing import List, Optional

import torch
from torch_geometric.data import Data

class TransformedSample(Data):
    def __init__(self, x: Optional[torch.Tensor] = None, edge_index: Optional[torch.Tensor] = None, edge_attr: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None, face: Optional[torch.Tensor] = None, original_pos: Optional[torch.Tensor] = None, pos_transf: Optional[torch.Tensor] = None, y_transf: Optional[torch.Tensor] = None, face_transf: Optional[torch.Tensor] = None, original_pos_transf: Optional[torch.Tensor] = None, transf = None):
        """
        Data paired with its random transform applied.

        :param x: Tensor that contains the original x
        :param edge_index: Tensor that contains the original edge_index
        :param edge_attr: Tensor that contains the original edge_attr
        :param y: Tensor that contains the original target
        :param pos: Tensor that contains the original pos
        :param face: Tensor that contains the original face
        :param pos_transf: Tensor that contains the transformation of pos
        """
        super(TransformedSample, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, face=face)
        self.original_pos = original_pos
        self.pos_transf = pos_transf
        self.y_transf = y_transf
        self.face_transf = face_transf
        self.original_pos_transf = original_pos_transf
        self.transf = transf