from typing import Optional, Any

import torch
import os
import os.path as osp
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import InMemoryDataset,Dataset, download_url
from torch_geometric.data import HeteroData
import pandas as pd
from tqdm import tqdm

import transform as tnf
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import random
import shutil
import pytorch_lightning as pl

from data import DailyData
random.seed(42)
torch.manual_seed(3407)
np.random.seed(0)



class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = T.Compose([tnf.ScaleEdges(attrs=["edge_attr"]),
                            T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                            T.ToUndirected(),
                            T.AddSelfLoops(),
                            tnf.RevDelete()])

    def prepare_data(self) -> None:
        # Download logic or first time prep
        # It is not recommended to assign state here (e.g. self.x = y).
        DailyData(self.data_dir,transform=self.transform)

    def setup(self, stage: Optional[str] = None) -> None:
        # data operations you might want to perform on every GPU

        data = DailyData(self.data_dir,transform=self.transform)
        self.metadata = data[0].metadata()
        n = (len(data) + 9) // 10
        if stage=="fit" or stage is None:
            self.train_data = data[:-2 * n]
            self.val_data = data[-2 * n: -n]
        if stage =="test" or stage is None:
            self.test_data = data[-n:]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        ...

    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     batch['x'] = transforms(batch['x'])
    #     return batch


if __name__ == '__main__':
    root = osp.join(os.getcwd(), "dailyroot")
    # proc_path = os.path.join(root, 'processed')
    # if os.path.exists(proc_path):
    #     shutil.rmtree(os.path.join(root,'processed'))
    #
    # if os.path.exists(proc_path):
    #     shutil.rmtree(os.path.join(root,'processed'))
    #
    # transform = T.Compose([tnf.ScaleEdges(attrs=["edge_attr"]),
    #                         T.NormalizeFeatures(attrs=["x", "edge_attr"]),
    #                         T.ToUndirected(),
    #                         T.AddSelfLoops(),
    #                         tnf.RevDelete()])
    #
    # data2 = DailyData(root, transform=transform)
    # nd2 = data2[1]

    data = GraphDataModule(root,5)
    print(data.metadata)
    #y= next(data.train_dataloader())