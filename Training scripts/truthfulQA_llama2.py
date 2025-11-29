###truthfulQA_llama
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch_geometric.loader import DataLoader,NeighborSampler
from torch_geometric.nn import GCNConv, global_mean_pool,GraphConv, global_max_pool,TopKPooling
import pickle
from data import train_data_Data,test_data_Data
from common_utils import get_roc_scores
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gnn", default='_cut',choices=["_cut",''])
parser.add_argument("--type", default='raw',choices=["screen",'raw'])
args = parser.parse_args()

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=0.5):
        super(GNNModel, self).__init__()
        torch.cuda.manual_seed(12345)
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, 64)
        self.classifier = torch.nn.Linear(64, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight,data.batch
        x = x.float()
        edge_weight = edge_weight.float()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2,training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2,training=self.training)
        x = global_mean_pool(x, batch)  # 
        x = self.classifier(x)
        return x


train_file_name = f'TruthfulQA/llama/{args.type}/TruthfulQA_llama_train_data_list.pkl'
test_file_name = f'TruthfulQA/llama/{args.type}/TruthfulQA_llama_test_data_list.pkl'

with open(train_file_name, 'rb') as file1:
    train_data = pickle.load(file1) 


with open(test_file_name, 'rb') as file2:
    test_data = pickle.load(file2)


train_data_list = train_data
test_data_list = test_data
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GNNModel(input_dim=4096, hidden_dim=256, output_dim=2).to(device)
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()



from sklearn.metrics import f1_score
def train():
    model.train()
    for epoch in range(30):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        currr_lr=scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Learning rate: {currr_lr}')


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    scores=[]
    label=[]
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        scores.append(pred.cpu())
        label.append(data.y.cpu())
        correct += pred.eq(data.y).sum().item()
    f1 = f1_score(label, scores)
    print(f"F1-Score: {f1:.4f}") 
    arc, acc, low = get_roc_scores(np.array(scores),np.array(label))
    return correct / len(loader.dataset)

for epoch in range(200):
    train()

torch.save(model.state_dict(), f'HaluGNN/model/truthfulQA_llama_gnn_model.pth')


model.load_state_dict(torch.load(f'HaluGNN/model/truthfulQA_llama_gnn_model.pth'))
train_acc = test(train_loader)
test_acc = test(test_loader)
print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

