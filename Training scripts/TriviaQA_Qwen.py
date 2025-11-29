###TriviaQA_Qwen
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch_geometric.loader import DataLoader,NeighborSampler
from torch_geometric.nn import GCNConv, global_mean_pool,GraphConv, global_max_pool,TopKPooling
import pickle
from torch_geometric.data import Data
from common_utils import get_roc_scores
import numpy as np
import argparse
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
matplotlib.use('Agg')



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
        self.features = {}

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight,data.batch
        x = x.float()
        self.features['raw'] = x.detach().cpu()
        edge_weight = edge_weight.float()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2,training=self.training)
        self.features['conv1'] = x.detach().cpu() 
 
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2,training=self.training)
        self.features['conv2'] = x.detach().cpu() 
        x = global_mean_pool(x, batch)  
        self.features['global_mean_pool'] = x.detach().cpu() 
        x = self.classifier(x)
        return x

train_file_name = f'TriviaQA/Qwen/{args.type}/TriviaQA_Qwen_train_data_list.pkl'
test_file_name = f'TriviaQA/Qwen/{args.type}/TriviaQA_Qwen_test_data_list.pkl'

with open(train_file_name, 'rb') as file1:
    train_data = pickle.load(file1)


with open(test_file_name, 'rb') as file2:

    test_data = pickle.load(file2)


train_data_list = train_data
test_data_list = test_data
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)   #train_data_list
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=True)    #test_data_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GNNModel(input_dim=3584, hidden_dim=256, output_dim=2).to(device)

optimizer = Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()



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


from sklearn.metrics import f1_score
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
        #print(pred,data.y)
        correct += pred.eq(data.y).sum().item()
    f1 = f1_score(label, scores)
    print(f"F1-Score: {f1:.4f}") 
    arc, acc, low = get_roc_scores(np.array(scores),np.array(label))
    return correct / len(loader.dataset)


for epoch in range(200):
    train()
torch.save(model.state_dict(), f'InternalStates/HaluGNN/model/TriviaQA_Qwen_gnn_model.pth')

model.load_state_dict(torch.load(f'InternalStates/HaluGNN/model/TriviaQA_Qwen_gnn_model.pth'))
train_acc = test(train_loader)
test_acc = test(test_loader)
print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


####Interpretability
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x, edge_index, edge_weight=None, batch=None, **kwargs):
        data = Data(x=x, edge_index=edge_index, 
                   edge_weight=edge_weight, 
                   batch=batch)
        return self.model(data)

wrapped_model = ModelWrapper(model)
explainer = Explainer(
    model=wrapped_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    )
)


def visualize_graph_explanation(data, edge_mask, threshold=0.1):

    fig, ax = plt.subplots(figsize=(12, 10))

    edge_index = data.edge_index.cpu().numpy().T
    edge_mask = edge_mask.cpu().detach().numpy()
    edge_weights = data.edge_weight.cpu().numpy() if hasattr(data, 'edge_weight') else np.ones(len(edge_index))
    
    G = nx.DiGraph()
    G.add_edges_from(
        (src, dst, {'weight': w, 'importance': imp}) 
        for (src, dst), w, imp in zip(edge_index, edge_weights, edge_mask)
    )
    
    pos = nx.spring_layout(G)
    
    edges_data = list(G.edges(data=True))
    important_edges = [e[:2] for e in edges_data if e[2]['importance'] > threshold]
    other_edges = [e[:2] for e in edges_data if e[2]['importance'] <= threshold]
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, ax=ax)
    sorted_nodes = sorted(G.nodes())
    nx.draw_networkx_labels(G, pos, labels={n:f"t{n}" for n in sorted_nodes}, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=other_edges,
        edge_color='gray', width=1, alpha=0.2, 
        arrows=True, ax=ax
    )
    
    edge_colors = [G.edges[e]['importance'] for e in important_edges]
    edges = nx.draw_networkx_edges(
        G, pos, edgelist=important_edges,
        edge_color=edge_colors, edge_cmap=plt.cm.Reds,
        width=3, arrows=True, arrowstyle='->', ax=ax
    )
    
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Reds, 
        norm=plt.Normalize(vmin=np.min(edge_mask), vmax=np.max(edge_mask))
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Edge Importance')
    
    ax.set_title(f"Graph Explanation (Threshold={threshold})")
    ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('output.png')



for data in test_loader: 
    data=data.to(device)
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_weight=getattr(data, 'edge_weight', None),
        batch=getattr(data, 'batch', None)
    )

    edge_mask = explanation.edge_mask.squeeze() 
    visualize_graph_explanation(data, edge_mask)
    break


# t-sne
def visualize_graph_embeddings(model, dataloader,layer):

    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _ = model(data) 
            
            graph_emb = model.features[layer].mean(dim=0).unsqueeze(0) 
            all_embeddings.append(graph_emb.cpu())
            all_labels.append(data.y.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    

    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(
    emb_2d[:, 0], 
    emb_2d[:, 1], 
    c=labels,               
    cmap='bwr',             
    s=50,                   
    edgecolors='k',          
    alpha=0.8,              
    linewidths=0.5        
)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color='blue', label='Class 0'),
        mpatches.Patch(color='red', label='Class 1')
    ]
    plt.legend(handles=legend_patches)

    plt.title('t-SNE Visualization (Binary Classification)')
    plt.colorbar(label='Class Probability', ticks=[0, 1]) 
    plt.show()
    plt.savefig(f'output_{layer}.png') 
