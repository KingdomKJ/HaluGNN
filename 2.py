###halueval_llama的gnn
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch_geometric.loader import DataLoader,NeighborSampler
from torch_geometric.nn import GCNConv, global_mean_pool,GraphConv, global_max_pool,TopKPooling
import pickle


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

# train_file_name = train_data_Data
# test_file_name = test_data_Data
train_file_name = f'InternalStates/LLM_Check_Hallucination_Detection-main/data/halueval/llama/screen/halueval_llama_train_data_list.pkl'
test_file_name = f'InternalStates/LLM_Check_Hallucination_Detection-main/data/halueval/llama/screen/halueval_llama_test_data_list.pkl'
# 使用with语句确保文件正确关闭
with open(train_file_name, 'rb') as file1:
    # 加载pkl文件内容
    train_data = pickle.load(file1)    #虽然无法查看，但可以正常读取
    #print(train_data)

# 使用with语句确保文件正确关闭
with open(test_file_name, 'rb') as file2:
    # 加载pkl文件内容
    test_data = pickle.load(file2)
    #print(test_data)

train_data_list = train_data
test_data_list = test_data
#加载训练集和测试集
# 使用邻居采样
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=True)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GNNModel(input_dim=4096, hidden_dim=256, output_dim=2).to(device)
# 设置优化器和学习率调度器
optimizer = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()


# 训练函数
def train():
    model.train()
    for epoch in range(30):
        for data in train_loader:
            #print(data.y)
            #print(data.x.type,data.edge_index.type,data.edge_weight.type,data.y.type)
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            #print(out,data.y)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        currr_lr=scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Learning rate: {currr_lr}')

# 测试函数
@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        print(model(data),pred)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

# # 训练和测试
# for epoch in range(200):
#train()
# 保存模型参数
#torch.save(model.state_dict(), f'InternalStates/LLM_Check_Hallucination_Detection-main/model/halueval_llama_gnn_model.pth')

# 加载模型参数
model.load_state_dict(torch.load(f'InternalStates/LLM_Check_Hallucination_Detection-main/model/halueval_llama_gnn_model.pth'))
train_acc = test(train_loader)
test_acc = test(test_loader)
print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')