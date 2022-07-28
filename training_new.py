
import os
import sys
from pathlib import Path

from data import DailyData

sys.path.append(str(Path(os.getcwd()).parents[1]))

import numpy as np
import pandas as pd
import torch
torch.manual_seed(3708)
from torch.nn.functional import mse_loss
from torch.nn import L1Loss
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from model_new import NetQtyModel
import transform as tnf


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')



def train_runner(model, optimizer, train_loader, device):
    """
    Model training method
    :param model:
    :return:
    """
    model.train()
    print("Going through train batches")
    total_examples = total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        preds = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict) # pred_sla,pred_netqty
        targets = data[('node1', 'to', 'node2')].edge_label.flatten().float().to(device)

        num_samples = preds.size(0)
        total_examples += num_samples
        loss = mse_loss(preds,targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_loss += float(loss) * num_samples

    return total_loss / total_examples

@torch.no_grad()
def val_runner(model, val_loader, device):
    """
    Conditional method to run train,eval modes
    :param mode:
    :return:
    """
    model.eval()
    total_examples = total_loss = 0
    for data in val_loader:
        data.to(device)
        preds = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)  # pred_sla,pred_netqty
        targets = data[('node1', 'to', 'node2')].edge_label.flatten().float().to(device)

        num_samples = preds.size(0)
        total_examples += num_samples
        loss = mse_loss(preds, targets)

        total_loss += float(loss) * num_samples

    return total_loss / total_examples


@torch.no_grad()
def test_runner(model, test_loader, device):
    """
    Conditional method to run train,eval modes
    :param mode:
    :return:
    """
    model.eval()
    total_examples = total_loss = 0
    for data in test_loader:
        data.to(device)
        preds = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)  # pred_sla,pred_netqty
        targets = data[('node1', 'to', 'node2')].edge_label.flatten().float().to(device)

        num_samples = preds.size(0)
        total_examples += num_samples
        loss = mse_loss(preds, targets)

        total_loss += float(loss) * num_samples

    return total_loss / total_examples


@torch.no_grad()
def test_eval(model,data,device):
    """
    Conditional method to run on test data
    :param mode:
    :return:
    """
    model.eval()
    data.to(device)
    preds = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)  # pred_sla,pred_netqty
    targets = data[('customer', 'orders', 'product')].edge_label.flatten().float().to(device)

    return preds,targets



if __name__=="__main__":
    ### Loading graph data from root directory
    #------------------------------------------
    transform = T.Compose([tnf.ScaleEdges(attrs=["edge_attr"]),
                           T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                           T.ToUndirected(),
                           T.AddSelfLoops(),
                           tnf.RevDelete()])
    root = os.path.join(os.getcwd(), "dailyroot")
    data = DailyData(root, transform=transform)

    n = (len(data) + 9) // 10
    train_data = data[:-2 * n]
    val_data = data[-2 * n: -n]
    test_data = data[-n:]

    batch_size = 5  # number of days/graphs to be taken in one batch
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    batch_size = 5 # number of days/graphs to be taken in one batch
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)



    ### Loading Model architecture
    #------------------------------------------
    train_data_batch = next(iter(train_loader)).to(device)
    test_data = next(iter(test_loader)).to(device)
    model = NetQtyModel(train_data_batch.metadata(), hidden_channels=512, out_channels=64, num_layers=2).to(device)


    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        # encoder = to_hetero(model.encoder,train_data.metadata(),aggr='sum')
        model(train_data_batch.x_dict, train_data_batch.edge_index_dict, train_data_batch.edge_attr_dict)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # cycle momentum needs to be False for Adam to work
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.3,step_size_up=10, cycle_momentum=False)


    ### Training Loop - 500 epochs
    #------------------------------------------
    # Initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    # Save train logs

    loss = model(train_data_batch.x_dict, train_data_batch.edge_index_dict, train_data_batch.edge_attr_dict)
    print(loss)


    for epoch in range(1,5):
        train_loss = train_runner(model, optimizer, train_loader, device)
        val_loss = val_runner(model,val_loader,device)
        test_loss = test_runner(model,test_loader,device)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, Test Loss: {test_loss:.4f}')

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        ## save the model if validation loss has decreased
        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min, val_loss))
            valid_loss_min = val_loss

    print("Stopped Training. Num-epochs reached")


