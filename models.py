import torch

import torch.nn.functional as F

from torch.nn import Linear, ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv


class GCNConvNet(torch.nn.Module):
    def __init__(self,
                 dataset,
                 hidden=[64],
                 dropout=0.5):
        super(GCNConvNet, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)

        self.out_fn = lambda x: F.log_softmax(x, dim=1)
        self.act_fn = ReLU()

        print(self)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return self.out_fn(x)