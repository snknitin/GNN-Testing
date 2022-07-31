from typing import Optional, Union, List

import torch
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
from torch.nn.functional import mse_loss
import collections


# from pytorch_lightning.metrics import tensor_metric
#
# @tensor_metric()
# def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#   return torch.sqrt(torch.mean(torch.pow(pred-target, 2.0)))

class GNNEncoder(pl.LightningModule):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
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


class EdgeDecoder(pl.LightningModule):
    def __init__(self, hidden_channels,out_channels):
        super().__init__()
        self.network = torch.nn.Sequential(
            Linear(-1, hidden_channels),
            LeakyReLU(),
            Dropout(p=0.2),
            Linear(hidden_channels, out_channels),
            LeakyReLU(),
            Dropout(p=0.2),
            Linear(out_channels, 32),
            LeakyReLU(),
            Dropout(p=0.2),
            Linear(32, 1)
        )



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
        z4 = self.network(z4)

        return z4.view(-1)


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
        return {"loss":loss}



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

    def predict_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        return preds




if __name__=="__main__":

    root = os.path.join(os.getcwd(), "dailyroot")
    model = NetQtyModel(root,hidden_channels=512, out_channels=64, num_layers=2,batch_size=5)
    # Enable chkpt , gpu, epochs
    trainer = pl.Trainer(fast_dev_run=True)
    # trainer = pl.Trainer(max_steps=1000,max_epochs=500,
    #                      auto_lr_find=True,gradient_clip_val=1.0,
    #                      accumulate_grad_batches=10
    #                      )
    trainer.fit(model)
    trainer.validate(model)
    # trainer.test(model)
    # trainer.predict(model)
