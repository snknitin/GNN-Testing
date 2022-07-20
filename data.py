import torch
import os
import os.path as osp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import InMemoryDataset,Dataset, download_url
from torch_geometric.data import HeteroData
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.loader import LinkNeighborLoader

import transform as tnf
import torch_geometric.transforms as T
import random
import shutil
random.seed(42)
torch.manual_seed(3407)
np.random.seed(0)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.root = root

    @property
    def raw_dir(self):
        return os.path.join(self.root,'raw')

    @property
    def raw_file_names(self):
        edge_dir = os.path.join(self.raw_dir,"relations")
        files = sorted(os.listdir(edge_dir))
        return [os.path.join(edge_dir,f) for f in files]

    @property
    def processed_file_names(self):
        return ['data_{0}.pt'.format(x) for x in range(30)]

    def download(self):
        # Download to `self.raw_dir`.
        ...

    def get_scalers(self):
        """
        Scaling the edge attributed based on the whole data for consistency
        we need to return scaler and perform the transform
        """

        def f(i):
            return pd.read_csv(i, low_memory=True)

        # Read edge_cols from all raw files
        df = pd.concat(map(f, self.raw_file_names))
        scaler = MinMaxScaler()
        scaler.fit(df)
        del df
        return scaler


    def load_full_node_csv(self, featpath, idxpath):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        df = pd.read_csv(featpath)
        map_df = pd.read_csv(idxpath)
        mapping = dict(zip(map_df["ent_name"], map_df["ent_idx"]))
        x = torch.tensor(df.values, dtype=torch.float)
        return x, mapping

    def load_edge_csv(self, edge_file_path, edge_cols, src_index_col, src_mapping,
                      dst_index_col, dst_mapping, encoders=None):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        df = pd.read_csv(edge_file_path)
        # src = [src_mapping[index] for index in df[src_index_col]]
        # dst = [dst_mapping[index] for index in df[dst_index_col]]
        src = []
        dst = []
        for index, row in df.iterrows():
            try:
                s = src_mapping[row[src_index_col]]
                d = dst_mapping[row[dst_index_col]]
            except:
                df.drop(index, inplace=True)
                # print("Missed a key")
                continue
            src.append(s)
            dst.append(d)

        # Updates edge indices for dailygraph based on node index to extract
        edge_index, src_extract, dst_extract = self.get_new_ids(src, dst)

        edge_attr = df[edge_cols]
        edge_attr = torch.tensor(edge_attr.values, dtype=torch.float)

        edge_label = None
        if encoders is not None:
            edge_label = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_label = torch.cat(edge_label, dim=-1)

        return src_extract, dst_extract, edge_index, edge_attr, edge_label

    def get_edges(self, x, y):
        src = np.random.randint(0, x, 350)
        dest = np.random.randint(0, y, 350)
        return [src, dest]

    def process(self):
        node_dir = osp.join(self.raw_dir, "node-features/")
        edge_dir = os.path.join(self.raw_dir, 'relations/')

        edge2to1scaler = self.get_scalers()
        edge3to2scaler = self.get_scalers()

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
    #transform = T.Compose([T.ToUndirected(), T.AddSelfLoops(), T.NormalizeFeatures(attrs=["x","edge_attr"])])


    data = DailyData(root)
    edge_stats = tnf.getfullstats(data)
    del data
    shutil.rmtree(os.path.join(root,'processed'))
    transform1 = T.Compose([T.ToUndirected(),
                          T.AddSelfLoops(),
                          tnf.ScaleEdges(stats=edge_stats, attrs=["edge_attr"]),
                          T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                          tnf.RevDelete()])

    transform2 = T.Compose([T.ToUndirected(),
                            T.AddSelfLoops(),
                            tnf.ScaleEdges(attrs=["edge_attr"]),
                            #T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                            tnf.RevDelete()])

    data = DailyData(root,transform=transform2)

    nd1 = data[0]
    print(nd1)
    print(nd1[nd1.edge_types[2]].edge_attr[0])
    print(nd1["node1"].x[0])
    # root = osp.join(os.getcwd(), "yearlyroot")
    # data1 = YearlyData(root)[0] # need to remove the [0] for loaders

    #data.to(device)
    # data1.to(device) # this doesn't work for daily
    print(data)

