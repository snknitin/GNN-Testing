import torch
import os
import os.path as osp
import numpy as np
from torch_geometric.data import InMemoryDataset,Dataset, download_url
from torch_geometric.data import HeteroData

from torch_geometric.loader import DataLoader
from torch_geometric.loader import LinkNeighborLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YearlyData(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['file1']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        ...

    def get_edges(self, x, y):
        src = np.random.randint(0, x, 3500)
        dest = np.random.randint(0, y, 3500)
        return [src, dest]

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        g1 = HeteroData()
        g1["node1"].x = torch.tensor(np.round(np.random.rand(1250, 6) * 10), dtype=torch.float)
        g1["node2"].x = torch.tensor(np.round(np.random.rand(2000, 6) * 20), dtype=torch.float)
        g1["node3"].x = torch.tensor(np.round(np.random.rand(100, 6) * 10), dtype=torch.float)

        g1['node2', 'to', 'node1'].edge_index = torch.tensor(self.get_edges(2000, 1250), dtype=torch.long)
        g1['node3', 'to', 'node2'].edge_index = torch.tensor(self.get_edges(100, 2000), dtype=torch.long)

        g1['node2', 'to', 'node1'].edge_attr = torch.tensor(np.round(np.random.rand(3500, 6) * 10), dtype=torch.float)
        g1['node3', 'to', 'node2'].edge_attr = torch.tensor(np.round(np.random.rand(3500, 5) * 10), dtype=torch.float)

        g1["node2", "to", "node1"].edge_label = torch.rand((3500, 1))
        g1["node3", "to", "node2"].edge_label = torch.rand((3500, 1))

        node_types, edge_types = g1.metadata()
        for node_type in node_types:
            g1[node_type].num_nodes = g1[node_type].x.size(0)

        data_list.append(g1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class DailyData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data_{0}.pt'.format(x) for x in range(30)]

    def download(self):
        # Download to `self.raw_dir`.
        ...
    def get_edges(self, x, y):
        src = np.random.randint(0, x, 350)
        dest = np.random.randint(0, y, 350)
        return [src, dest]

    def process(self):
        idx = 0
        for _ in range(30):
            # Read data from `raw_path`.
            data = HeteroData()
            data["node1"].x = torch.tensor(np.round(np.random.rand(125, 6) * 10), dtype=torch.float)
            data["node2"].x = torch.tensor(np.round(np.random.rand(200, 6) * 20), dtype=torch.float)
            data["node3"].x = torch.tensor(np.round(np.random.rand(10, 6) * 10), dtype=torch.float)

            data['node2', 'to', 'node1'].edge_index = torch.tensor(self.get_edges(200, 125), dtype=torch.long)
            data['node3', 'to', 'node2'].edge_index = torch.tensor(self.get_edges(10, 200), dtype=torch.long)

            data['node2', 'to', 'node1'].edge_attr = torch.tensor(np.round(np.random.rand(350, 6) * 10),
                                                                dtype=torch.float)
            data['node3', 'to', 'node2'].edge_attr = torch.tensor(np.round(np.random.rand(350, 5) * 10),
                                                                dtype=torch.float)

            data["node2", "to", "node1"].edge_label = torch.rand((350, 1))
            data["node3", "to", "node2"].edge_label = torch.rand((350, 1))

            node_types, edge_types = data.metadata()
            for node_type in node_types:
                data[node_type].num_nodes = data[node_type].x.size(0)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


if __name__ == '__main__':
    root = osp.join(os.getcwd(), "dailyroot")
    data1 = DailyData(root)

    root = osp.join(os.getcwd(), "yearlyroot")
    data = YearlyData(root)[0] # need to remove the [0] for loaders

    #data.to(device)
    # data1.to(device) # this doesn't work

    idxs= list(data.edge_index_dict.items())
    #loader = DataLoader(data, batch_size=4)
    loader = LinkNeighborLoader(data,
                                num_neighbors={key: [30] * 2 for key in data.edge_types},
                                edge_label_index=[data.edge_types[0],data[data.edge_types[0]].edge_index],
                                edge_label=data[data.edge_types[0]].edge_label,
                                batch_size=1024)
    sampled_hetero_data = next(iter(loader))
    print(sampled_hetero_data)