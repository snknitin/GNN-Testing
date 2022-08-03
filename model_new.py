from typing import Optional, Union, List
import torch
from torch.nn import Linear, ReLU, BatchNorm1d, ModuleList, L1Loss, LeakyReLU, Dropout, MSELoss
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS, EPOCH_OUTPUT
import torch_geometric.transforms as T
import transform as tnf
import os
from models import GNNEncoder,EdgeDecoder
from data_new import GraphDataModule
import hydra
import omegaconf
import pyrootutils





class NetQtyModel(pl.LightningModule):
    def __init__(self,encoder,decoder):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.transform = T.Compose([tnf.ScaleEdges(attrs=["edge_attr"]),
                                    T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                                    T.ToUndirected(),
                                    T.AddSelfLoops(),
                                    tnf.RevDelete()])

        self.encoder = self.hparams.encoder
        self.decoder = self.hparams.decoder


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
        self.log("val_loss",avg_val_loss,on_epoch=True,prog_bar=True)
        return {'val_loss':avg_val_loss}



    def test_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        targets = batch[('node1', 'to', 'node2')].edge_label.flatten().float()
        test_loss = self.loss_function(preds,targets)
        self.log("test_loss", test_loss)
        return {"val_loss":test_loss}

    def _shared_eval(self, batch, batch_idx, stage):
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        targets = batch[('node1', 'to', 'node2')].edge_label.flatten().float()
        loss = self.loss_function(preds, targets)
        self.log(f"{stage}_loss", loss,on_step=False,on_epoch=True,prog_bar=True)
        return {f"{stage}_loss":loss}

    def predict_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        return preds




if __name__=="__main__":

    data_dir = os.path.join(os.getcwd(), "dailyroot")
    root = pyrootutils.setup_root(__file__, pythonpath=True)
    data_cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "dailydata.yaml")
    data = hydra.utils.instantiate(data_cfg)
    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "net_qty_model.yaml")
    model= hydra.utils.instantiate(model_cfg)
    # Enable chkpt , gpu, epochs
    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=model,datamodule=data)
    # trainer = pl.Trainer(max_steps=1000,max_epochs=500,check_val_every_n_epochs=10,
    #                      auto_lr_find=True,gradient_clip_val=1.0,deterministic=True,
    #                      accumulate_grad_batches=4,sync_batchnorm=True
    #                      )

    # trainer.validate(model)
    # trainer.test(model)
    # trainer.predict(model)
