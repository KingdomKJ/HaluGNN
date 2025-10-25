
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch_geometric.loader import DataLoader,NeighborSampler
from torch_geometric.nn import GCNConv, global_mean_pool,GraphConv, global_max_pool,TopKPooling
import pickle
from common_utils import get_roc_scores
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gnn", default='_cut',choices=["_cut",''])
parser.add_argument("--type", default='screen',choices=["screen",'raw'])
args = parser.parse_args()


class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=0.5):
        super(GNNModel, self).__init__()
        torch.cuda.manual_seed(12345)
        self.conv1 = GraphConv(input_dim, hidden_dim)
        #self.pool = TopKPooling(hidden_dim, ratio=k)
        self.conv2 = GraphConv(hidden_dim, 64)
        self.classifier = torch.nn.Linear(64, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight,data.batch
        x = x.float()
        edge_weight = edge_weight.float()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2,training=self.training)
        #x, edge_index, _, batch, _, _ = self.pool(x, edge_index, edge_weight,batch)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2,training=self.training)
        x = global_mean_pool(x, batch)  # 
        # x = F.dropout(x, p=0.5,training=self.training)
        x = self.classifier(x)
        return x

