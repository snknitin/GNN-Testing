import torch
from torch.nn.functional import leaky_relu, dropout
from torch.nn import Linear,ReLU, BatchNorm1d, ModuleList,L1Loss
from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero,HeteroConv,GATConv,GINEConv

import collections


class GNNEncoder(torch.nn.Module):
    def __init__(self, metadata,hidden_channels, out_channels, num_layers):
        super().__init__()

        layers = (hidden_channels, out_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), layers[i])
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: leaky_relu(x) for key, x in x_dict.items()}

        return x_dict


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels,out_channels):
        super().__init__()
        self.lin1 = Linear(-1, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.lin3 = Linear(out_channels, 32)
        self.lin4 = Linear(32, 1)


    def forward(self, z_dict, edge_index_dict,edge_attr_dict):
        ship, cust = edge_index_dict[('node1', 'to', 'node2')]
        cust,prod = edge_index_dict[('node2', 'to', 'node3')]
        attr1 = edge_attr_dict[('node1', 'to', 'node2')]
        attr2 = edge_attr_dict[('node2', 'to', 'node3')]
        # Concatenate the embeddings and edge attributed
        z1 = leaky_relu(z_dict['node1'][ship])
        z2 = leaky_relu(z_dict['node2'][cust])
        z3 = leaky_relu(z_dict['node3'][prod])

        z4 = leaky_relu(torch.cat([z1,z2,z3,attr1,attr2], dim=-1))
        z4 = dropout(z4, p=0.2, training=self.training)

        z4 = self.lin1(z4)
        z4 = leaky_relu(z4)
        z4 = dropout(z4, p=0.2, training=self.training)
        z4 = self.lin2(z4)
        z4 = leaky_relu(z4)
        z4 = dropout(z4, p=0.2, training=self.training)
        z4 = self.lin3(z4)
        z4 = leaky_relu(z4)
        z4 = self.lin4(z4)

        return z4.view(-1)


class NetQtyModel(torch.nn.Module):
    def __init__(self, metadata,hidden_channels,out_channels, num_layers):
        super().__init__()
        self.encoder = GNNEncoder(metadata, hidden_channels, out_channels, num_layers)
        self.decoder = EdgeDecoder(hidden_channels,out_channels)

    def forward(self, x_dict,edge_index_dict,edge_attr_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict,edge_index_dict,edge_attr_dict)





if __name__=="__main__":
    pass