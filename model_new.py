from typing import Optional, Union, List

import torch
from pytorch_lightning.utilities import argparse
from torch.nn.functional import leaky_relu, dropout
from torch.nn import Linear, ReLU, BatchNorm1d, ModuleList, L1Loss, LeakyReLU, Dropout, MSELoss
from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero,HeteroConv,GATConv,GINEConv
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS, EPOCH_OUTPUT
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import transform as tnf
import os
from data import DailyData
from models import GNNEncoder,EdgeDecoder
from torch.nn.functional import mse_loss
import collections




class NetQtyModel(pl.LightningModule):
    def __init__(self, data_dir,hidden_channels,out_channels, num_layers,batch_size):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = T.Compose([tnf.ScaleEdges(attrs=["edge_attr"]),
                                    T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                                    T.ToUndirected(),
                                    T.AddSelfLoops(),
                                    tnf.RevDelete()])

        self.prepare_data()
        self.setup()

        # self.metadata = (['node1', 'node2', 'node3'],
        #                  [('node2', 'to', 'node3'),
        #                   ('node1', 'to', 'node2'),
        #                   ('node3', 'rev_to', 'node2'),
        #                   ('node2', 'rev_to', 'node1')])

        # model
        self.encoder = GNNEncoder(self.metadata, hidden_channels, out_channels, num_layers)
        self.decoder = EdgeDecoder(hidden_channels,out_channels)


    def forward(self, x_dict,edge_index_dict,edge_attr_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict,edge_index_dict,edge_attr_dict)

    def loss_function(self,pred,targets):
        loss = MSELoss()
        return loss(pred,targets)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # cycle momentum needs to be False for Adam to work
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.3, step_size_up=10,
                                                      cycle_momentum=False)
        return [optimizer], [lr_scheduler]

    def prepare_data(self) -> None:
        DailyData(self.data_dir, transform=self.transform)

    def setup(self, stage: Optional[str] = None) -> None:
        data = DailyData(self.data_dir,transform=self.transform)
        self.metadata = data[0].metadata()
        n = (len(data) + 9) // 10
        if stage in (None, "fit"):
            self.train_data = data[:-2 * n]
            self.val_data = data[-2 * n: -n]
        if stage in (None, "test"):
            self.test_data = data[-n:]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)


    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        targets = batch[('node1', 'to', 'node2')].edge_label.flatten().float()
        loss = self.loss_function(preds,targets)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def training_epoch_end(self, training_step_outputs: EPOCH_OUTPUT) -> None:
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        #return {'loss': avg_loss}

    def validation_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        results = self.training_step(batch,batch_idx)
        return results

    def validation_epoch_end(self, val_step_outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        # [results,results,results ...]
        avg_val_loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        return {'val_loss':avg_val_loss}



    def test_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        targets = batch[('node1', 'to', 'node2')].edge_label.flatten().float()
        test_loss = self.loss_function(preds,targets)
        self.log("test_loss", test_loss)
        return {"val_loss":test_loss}

    def _shared_eval(self, batch, batch_idx, prefix):
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        targets = batch[('node1', 'to', 'node2')].edge_label.flatten().float()
        loss = self.loss_function(preds, targets)
        self.log(f"{prefix}_loss", loss)

    def predict_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        return preds




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true", default=False, help="Whether to use GPU in the cloud")
    hparams = parser.parse_args()
    root = os.path.join(os.getcwd(), "dailyroot")
    model = NetQtyModel(root,hidden_channels=512, out_channels=64, num_layers=2,batch_size=5)
    # Enable chkpt , gpu, epochs
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    # trainer = pl.Trainer(max_steps=1000,max_epochs=500,check_val_every_n_epochs=10,
    #                      auto_lr_find=True,gradient_clip_val=1.0,deterministic=True,
    #                      accumulate_grad_batches=4,sync_batchnorm=True
    #                      )

    trainer.validate(model)
    # trainer.test(model)
    # trainer.predict(model)
