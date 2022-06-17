import torch_geometric.transforms as T
from collections import defaultdict
import torch
from torch_geometric.loader import DataLoader

from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('scale_edges')
class ScaleEdges(BaseTransform):
    r"""Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
    (functional name: :obj:`normalize_features`).

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(self, stats, attrs: List[str] = ["edge_attr"]):
        self.stats = stats
        self.attrs = attrs

    def __call__(self, data: Union[Data, HeteroData]):
        for edge,store in list(zip(data.edge_types,data.edge_stores)):
            edge_stats = self.stats[edge]
            for key, value in store.items(*self.attrs):
                value = value - value.min()
                xmean = edge_stats["mean"]
                xstd = edge_stats["std"]
                xmin, xmax = edge_stats["min"], edge_stats["max"]
                #value.div_(value.sum(dim=-1, keepdim=True).clamp_(min=1.))
                store[key] = (value-xmean).div(xmax-xmin)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def getfullstats(dailydata):
    """
    Pass the data into a loader to get the full stats
    Do this before Transforms that add reverse edges
    Create a transform for this if you want to limit edge columns
    """
    loader = DataLoader(dailydata, batch_size=len(dailydata))
    data = next(iter(loader))
    edge_stats = defaultdict(dict)
    for edge in data.edge_types:
        e = data[edge].edge_attr
        # get stats
        edge_stats[edge]["mean"] = e.mean(dim=0, keepdim=True)
        edge_stats[edge]["std"] = e.std(dim=0, keepdim=True)
        edge_stats[edge]["min"] = torch.min(e, 0).values
        edge_stats[edge]["max"] = torch.max(e, 0).values
    return edge_stats

def edge_norm_fn(data):
    """
    Normalization of edge_attributes
    """
    edge_stats = getfullstats(data)
    for edge in data[0].edge_types:
        xmean = edge_stats[edge]["mean"]
        xstd = edge_stats[edge]["std"]
        xmin,xmax = edge_stats[edge]["min"],edge_stats[edge]["max"]
        for i in range(len(data)):
            x = data[i][edge].edge_attr
            data[i][edge].edge_attr = (x-xmean)/(xmax-xmin)
    return data
