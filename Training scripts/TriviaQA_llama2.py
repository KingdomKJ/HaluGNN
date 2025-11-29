###TriviaQA_llama的gnn，是不是因为数据分布平衡才导致效果比较好的？
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
matplotlib.use('Agg')  # 无界面后端
# plt.rcParams['font.family'] = 'Times New Roman'          # 设置全局字体为 Arial
# plt.rcParams['font.size'] = 14                   # 设置全局字体大小为 14


parser = argparse.ArgumentParser()
parser.add_argument("--gnn", default='_cut',choices=["_cut",''])
parser.add_argument("--type", default='raw',choices=["screen",'raw'])
args = parser.parse_args()
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=0.5):
        super(GNNModel, self).__init__()
        torch.cuda.manual_seed(12345)
        self.conv1 = GraphConv(input_dim, hidden_dim)
        #self.pool = TopKPooling(hidden_dim, ratio=k)
        self.conv2 = GraphConv(hidden_dim, 64)
        self.classifier = torch.nn.Linear(64, output_dim)

        # 用于存储中间特征
        self.features = {}

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight,data.batch
        x = x.float()
        self.features['raw'] = x.detach().cpu()  # 保存第一层输出
        edge_weight = edge_weight.float()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2,training=self.training)
        self.features['conv1'] = x.detach().cpu()  # 保存第一层输出
        ##x, edge_index, _, batch, _, _ = self.pool(x, edge_index, edge_weight,batch)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2,training=self.training)
        self.features['conv2'] = x.detach().cpu()  # 保存第二层输出
        x = global_mean_pool(x, batch)  # 
        self.features['global_mean_pool'] = x.detach().cpu()  # 保存第二层输出
        # x = F.dropout(x, p=0.5,training=self.training)
        x = self.classifier(x)
        return x

# train_file_name = train_data_Data
# test_file_name = test_data_Data

train_file_name = f'/data2/kklg/internalstates_data/TriviaQA/llama/{args.type}/TriviaQA_llama_train_data_list.pkl'
test_file_name = f'/data2/kklg/internalstates_data/TriviaQA/llama/{args.type}/TriviaQA_llama_test_data_list.pkl'
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
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)   #train_data_list
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)    #test_data_list

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
from sklearn.metrics import f1_score
@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    scores=[]
    label=[]
    for data in loader:
        data = data.to(device)
        #print(data.x.shape)
        pred = model(data).argmax(dim=1)
        scores.append(pred.cpu())
        label.append(data.y.cpu())
        #print(pred,data.y)
        correct += pred.eq(data.y).sum().item()
    #print(label.count(1))
    f1 = f1_score(label, scores)
    print(f"F1-Score: {f1:.4f}") 
    arc, acc, low = get_roc_scores(np.array(scores),np.array(label))
    return correct / len(loader.dataset)

# # 训练和测试
# for epoch in range(200):
# train()
# # 保存模型参数
# torch.save(model.state_dict(), f'InternalStates/HaluGNN/model/TriviaQA_llama_gnn_model.pth')

# 加载模型参数
model.load_state_dict(torch.load(f'InternalStates/HaluGNN/model/TriviaQA_llama_gnn_model.pth'))
# train_acc = test(train_loader)
# test_acc = test(test_loader)
# print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

import time
#####统计推理一条的时间
for data in test_loader:
    data = data.to(device)
    #print(data.x.shape)
    start_time = time.time()
    pred = model(data).argmax(dim=1)
    end_time = time.time()
    print(f"End-to-end time: {(end_time - start_time) * 1000:.4f} ms")
    break

# ##统计参数数量
# from torch_geometric.data import Data
# from torchinfo import summary
# from torch_geometric.nn import summary as gnn_summary
# for data in test_loader:
#     data = data.to(device)
#     print(data)
#     x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight,data.batch
#     data1 = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
#     #input_data = (x, edge_index, edge_weight, batch)
#     summary(model,data1)
#     print(gnn_summary(model, data1) )# 會給出每層的參數量和總量
#     break
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total parameters: {total_params:,}")



############可解释性
# # 2. 模型包装器
# class ModelWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def forward(self, x, edge_index, edge_weight=None, batch=None, **kwargs):
#         data = Data(x=x, edge_index=edge_index, 
#                    edge_weight=edge_weight, 
#                    batch=batch)
#         return self.model(data)

# # 3. 初始化模型和解释器
# wrapped_model = ModelWrapper(model)
# explainer = Explainer(
#     model=wrapped_model,
#     algorithm=GNNExplainer(epochs=200),
#     explanation_type='model',
#     node_mask_type='attributes',
#     edge_mask_type='object',
#     model_config=dict(
#         mode='binary_classification',
#         task_level='graph',
#         return_type='raw',
#     )
# )


# # 4. 修正后的可视化函数,原图
# def visualize_graph(data,  count,threshold=0.2):
#     # 根据期刊要求设置figsize
#     fig_width_cm = 17  # 全版图宽度
#     fig_height_cm = 15  # 高度
#     fig_width_inch = fig_width_cm * 0.3937
#     fig_height_inch = fig_height_cm * 0.3937

#     fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), constrained_layout=True)
#     # 创建图形和轴对象
#     #fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
    
#     # 数据转换
#     edge_index = data.edge_index.cpu().numpy().T
#     edge_weights = data.edge_weight.cpu().numpy() if hasattr(data, 'edge_weight') else np.ones(len(edge_index))
    
#     # 创建图结构（改用更高效的添加边方式）
#     G = nx.DiGraph()
#     G.add_edges_from(
#         (src, dst, {'weight': w, }) 
#         for (src, dst), w in zip(edge_index, edge_weights)
#     )
    
#     # 计算布局
#     pos = nx.spring_layout(G)
    
#     # 边筛选（完全避免使用np.alltrue）
#     edges_data = list(G.edges(data=True))
#     important_edges = [e[:2] for e in edges_data if e[2]['weight'] > threshold]
#     other_edges = [e[:2] for e in edges_data if e[2]['weight'] <= threshold]
    
#     # 绘图（全部显式传递ax参数）
#     nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=250, ax=ax)
#     sorted_nodes = sorted(G.nodes())
#     #print(sorted_nodes)
#     nx.draw_networkx_labels(G, pos, labels={n:'t$_{{ {} }}$'.format(n) for n in sorted_nodes}, ax=ax,font_family='Arial',font_size=10)
#     #print({n:'t$_{{ {} }}$'.format(n) for n in sorted_nodes})
#     nx.draw_networkx_edges(
#         G, pos, edgelist=other_edges,
#         edge_color='gray', width=1, alpha=0.2, 
#         arrows=True, ax=ax
#     )
    
#     edge_colors = [G.edges[e]['weight'] for e in important_edges]
#     # 使用新的安全绘制方法
#     edges = nx.draw_networkx_edges(
#         G, pos, edgelist=important_edges,
#         edge_color=edge_colors, edge_cmap=plt.cm.Reds,
#         width=3, arrows=True, arrowstyle='->', ax=ax
#     )
    
#     # 颜色条处理
#     sm = plt.cm.ScalarMappable(
#         cmap=plt.cm.Reds, 
#         norm=plt.Normalize(vmin=np.min(edge_weights), vmax=np.max(edge_weights))
#     )
#     sm.set_array([])
    
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="3%", pad=0.05)

#     # 添加colorbar
#     cbar = fig.colorbar(sm, cax=cax)
#     cbar.ax.tick_params(labelsize=8,labelfontfamily='Arial')  #设置色标刻度字体大小。
#     cbar.set_ticks([0,0.2,0.4,0.6,0.8,1])
#     font = {'family' : 'Times New Roman',
#             'size'   : 14,
#             }
#     cbar.set_label('Edge Weights',fontdict=font) #设置colorbar的标签字体及其大小
#     #fig.colorbar(sm, ax=ax, label='Edge Weights', pad=0.03)
    
#     ax.set_title(f"Original Graph",fontdict=font)  #(Threshold={threshold})
#     ax.axis('on')

#     #plt.tight_layout()
#     plt.show()
#     plt.savefig(f'origin_graph_{count}.pdf', bbox_inches='tight')  # 手动保存

# # 4. 解释与可视化函数
# def explain_nodes(data, node_mask,edge_mask, threshold=0.15):
#     plt.figure(figsize=(12, 10))

      
#     # 1. 处理批次数据（确保只分析单个图）
#     if hasattr(data, 'batch'):
#         batch_mask = (data.batch == 0)  # 假设处理第一个图
#         node_mask = node_mask[batch_mask]
#         edge_mask = edge_mask[:data.edge_index.size(1)][batch_mask[data.edge_index[0]] & batch_mask[data.edge_index[1]]]
    
#     # 2. 创建有向图
#     G = nx.DiGraph()
#     edge_index = data.edge_index.cpu().numpy().T
#     edge_weights = data.edge_weight.cpu().numpy()
    
#     # 添加边（确保维度匹配）
#     for i, (src, dst) in enumerate(edge_index):
#         if not hasattr(data, 'batch') or (batch_mask[src] and batch_mask[dst]):
#             G.add_edge(src, dst, weight=edge_weights[i], importance=edge_mask[i].item())
#     print(node_mask.shape)
#     # 3. 添加节点属性（关键修正）
#     node_mask=node_mask.cpu()
#     node_colors = torch.norm(node_mask, dim=1)   # 确保一维 [num_nodes]
#     print(node_colors.shape)
#     # 归一化到[0,1]范围
#     node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())
    
#     # 4. 验证维度
#     assert len(node_colors) == data.num_nodes, f"颜色数{len(node_colors)}≠节点数{data.num_nodes}"
#     assert node_colors.ndim == 1, "颜色数据必须是一维"

#     # 5. 可视化
#     pos = nx.spring_layout(G, seed=42)
#     sorted_nodes = sorted(G.nodes())
#     nx.draw_networkx_labels(G, pos, labels={n:f"t{n}" for n in sorted_nodes})
#     nodes = nx.draw_networkx_nodes(
#         G, pos,
#         node_color=node_colors,  # 一维数组 [num_nodes]
#         cmap=plt.cm.viridis,
#         node_size=200,
#         alpha=0.8,
#     )
#     nx.draw_networkx_edges(
#         G, pos, 
#         edge_color='gray', 
#         width=0.5,
#         alpha=0.2,
#         arrows=True,
#         arrowstyle='-|>',
#         arrowsize=10,
#     )
#     plt.colorbar(nodes, label='Node Importance')  # 关联到ax1
#     plt.title(f'Important Nodes')
#     plt.savefig('node_explanation.png', dpi=300)
#     plt.show()


# def explain_subgraph(data, node_mask,edge_mask, threshold=0.15):
#     plt.figure(figsize=(12, 10))
    
#     # 1. 处理批次数据（确保只分析单个图）
#     if hasattr(data, 'batch'):
#         batch_mask = (data.batch == 0)  # 假设处理第一个图
#         node_mask = node_mask[batch_mask]
#         edge_mask = edge_mask[:data.edge_index.size(1)][batch_mask[data.edge_index[0]] & batch_mask[data.edge_index[1]]]
    
#     # 2. 创建有向图
#     G = nx.DiGraph()
#     edge_index = data.edge_index.cpu().numpy().T
#     edge_weights = data.edge_weight.cpu().numpy()
    
#     # 筛选重要边和节点
#     important_edges = [e for e in G.edges() if G.edges[e]['importance'] > threshold]
#     important_nodes = set()
#     for e in important_edges:
#         important_nodes.update(e)
    
#     pos = nx.spring_layout(G, seed=42)
#     # 绘制重要子图
#     subgraph = G.edge_subgraph(important_edges)
#     node_colors = ['red' if n in important_nodes else 'lightgray' for n in subgraph.nodes()]
#     edge_colors = ['blue' if G.edges[e]['importance'] > threshold else 'lightgray' 
#                   for e in subgraph.edges()]
    
#     nx.draw_networkx(
#         subgraph, pos,
#         node_color=node_colors,
#         edge_color=edge_colors,
#         width=1.5,
#         arrows=True,
#         arrowstyle='-|>',
#         arrowsize=10,
#         with_labels=True,
#         font_size=8
#     )
#     plt.title(f'Important Subgraph (Threshold={threshold})')
    
#     plt.tight_layout()
#     plt.savefig('subgraph_explanation.png', dpi=300)
#     plt.show()


# def visualize_edge_explanation(data, edge_mask, threshold=0.15):
#     # 创建图形和轴对象
#     fig, ax = plt.subplots(figsize=(15, 10))
    
    
#     # 数据转换
#     edge_index = data.edge_index.cpu().numpy().T
#     edge_mask = edge_mask.cpu().detach().numpy()
#     edge_weights = data.edge_weight.cpu().numpy() if hasattr(data, 'edge_weight') else np.ones(len(edge_index))
    
#     # 创建图结构（改用更高效的添加边方式）
#     G = nx.DiGraph()
#     G.add_edges_from(
#         (src, dst, {'weight': w, 'importance': imp}) 
#         for (src, dst), w, imp in zip(edge_index, edge_weights, edge_mask)
#     )
    
#     # 计算布局
#     pos = nx.spring_layout(G)
    
#     # 边筛选（完全避免使用np.alltrue）
#     edges_data = list(G.edges(data=True))
#     important_edges = [e[:2] for e in edges_data if e[2]['importance'] > threshold]
#     other_edges = [e[:2] for e in edges_data if e[2]['importance'] <= threshold]
    
#     # 绘图（全部显式传递ax参数）
#     nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, ax=ax)
#     sorted_nodes = sorted(G.nodes())
#     nx.draw_networkx_labels(G, pos, labels={n:f"t{n}" for n in sorted_nodes}, ax=ax)
#     # 不画重要性低于0.1的
#     # nx.draw_networkx_edges(
#     #     G, pos, edgelist=other_edges,
#     #     edge_color='gray', width=1, alpha=0.2, 
#     #     arrows=True, ax=ax
#     # )
    
#     edge_colors = [G.edges[e]['importance'] for e in important_edges]
#     # 使用新的安全绘制方法
#     edges = nx.draw_networkx_edges(
#         G, pos, edgelist=important_edges,
#         edge_color=edge_colors, edge_cmap=plt.cm.Reds,
#         width=3, arrows=True, arrowstyle='->', ax=ax
#     )
    
#     # 颜色条处理
#     sm = plt.cm.ScalarMappable(
#         cmap=plt.cm.Reds, 
#         norm=plt.Normalize(vmin=np.min(edge_mask), vmax=np.max(edge_mask))
#     )
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, label='Edge Importance')
    
#     ax.set_title(f"Edge Explanation (Threshold={threshold})")
#     ax.axis('on')

#     plt.tight_layout()
#     plt.show()
#     plt.savefig('edge_explanation.png', dpi=300, bbox_inches='tight')  # 手动保存

# def node_edge_explanation(data, node_mask, edge_mask,count, threshold=0.15):
#     # 创建图形和轴对象
#     # 根据期刊要求设置figsize
#     fig_width_cm = 17  # 全版图宽度
#     fig_height_cm = 15  # 高度
#     fig_width_inch = fig_width_cm * 0.3937
#     fig_height_inch = fig_height_cm * 0.3937

#     fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), constrained_layout=True)

    
#     # 数据转换
#     edge_index = data.edge_index.cpu().numpy().T
#     edge_mask = edge_mask.cpu().detach().numpy()
    
#     edge_weights = data.edge_weight.cpu().numpy() if hasattr(data, 'edge_weight') else np.ones(len(edge_index))
    
    


#     # 创建图结构（改用更高效的添加边方式）
#     G = nx.DiGraph()
#     G.add_edges_from(
#         (src, dst, {'weight': w, 'importance': imp}) 
#         for (src, dst), w, imp in zip(edge_index, edge_weights, edge_mask)
#     )
    
#     # 计算布局
#     pos = nx.spring_layout(G, seed=42)
    

#     # 边筛选（完全避免使用np.alltrue）
#     edges_data = list(G.edges(data=True))
#     important_edges = [e[:2] for e in edges_data if e[2]['importance'] > threshold]
#     other_edges = [e[:2] for e in edges_data if e[2]['importance'] <= threshold]
    
#     #########节点
#     # 绘图（全部显式传递ax参数）
#     node_mask=node_mask.cpu()
#     node_colors = torch.norm(node_mask, dim=1)   # 确保一维 [num_nodes]
#     print(node_colors.shape)
#     # 归一化到[0,1]范围
#     node_colors = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())
    
#     # 4. 验证维度
#     assert len(node_colors) == data.num_nodes, f"颜色数{len(node_colors)}≠节点数{data.num_nodes}"
#     assert node_colors.ndim == 1, "颜色数据必须是一维"

#     # 5. 可视化
#     sorted_nodes = sorted(G.nodes())
#     nx.draw_networkx_labels(G, pos, labels={n:'t$_{{ {} }}$'.format(n) for n in sorted_nodes}, ax=ax,font_family='Arial',font_size=10)
#     nodes = nx.draw_networkx_nodes(
#         G, pos,
#         node_color=node_colors,  # 一维数组 [num_nodes]
#         cmap=plt.cm.viridis,
#         node_size=250,
#         alpha=0.8,ax=ax
#     )
#     # 不画重要性低于0.1的
#     # nx.draw_networkx_edges(
#     #     G, pos, edgelist=other_edges,
#     #     edge_color='gray', width=1, alpha=0.2, 
#     #     arrows=True, ax=ax
#     # )
    
#     edge_colors = [G.edges[e]['importance'] for e in important_edges]
#     # 使用新的安全绘制方法
#     edges = nx.draw_networkx_edges(
#         G, pos, edgelist=important_edges,
#         edge_color=edge_colors, edge_cmap=plt.cm.Reds,
#         width=3, arrows=True, arrowstyle='->', ax=ax
#     )
    
#     # 颜色条处理
#     sm = plt.cm.ScalarMappable(
#         cmap=plt.cm.Reds, 
#         norm=plt.Normalize(vmin=np.min(edge_mask), vmax=np.max(edge_mask))
#     )
#     sm.set_array([])
#     divider = make_axes_locatable(ax)
#     cax1 = divider.append_axes("right", size="3%", pad=0.05)
#     cax2 = divider.append_axes("left", size="3%", pad=0.05)
    
#     # 添加colorbar
#     cbar1 = fig.colorbar(sm, cax=cax1)
#     cbar2=fig.colorbar(nodes, cax=cax2)
    
#     # 调整colorbar的刻度显示在外侧
#     cbar2.ax.yaxis.set_ticks_position('left')  # 刻度显示在右侧
#     cbar2.ax.yaxis.set_label_position('left')  # label显示在右侧
#     cbar1.ax.tick_params(labelsize=8,labelfontfamily='Arial')  #设置色标刻度字体大小。
#     cbar2.ax.tick_params(labelsize=8,labelfontfamily='Arial')  #设置色标刻度字体大小。
#     #cbar1.set_ticks([0,0.2,0.4,0.6,0.8,1])
#     cbar2.set_ticks([0,0.2,0.4,0.6,0.8,1])
#     font = {'family' : 'Times New Roman',
#             'size'   : 14,
#             }
    
#     ax.set_title(f"Original Graph",fontdict=font)  #(Threshold={threshold})
#     # 设置colorbar的label
#     cbar1.set_label('Edge Importance', labelpad=10,fontdict=font)
#     cbar2.set_label('Node Importance', labelpad=10,fontdict=font)
#     # fig.colorbar(nodes, ax=ax, label='Node Importance',location='left', pad=0.025)  #shrink=0.5,
#     # fig.colorbar(sm, ax=ax, label='Edge Importance',location='right', pad=0.03)  #shrink=0.5,

#     ax.set_title(f"Interpretability Analysis",fontdict=font)
#     ax.axis('on')

#     #plt.tight_layout()
#     plt.show()
#     plt.savefig(f'node&edge_explanation_{count}.pdf', bbox_inches='tight')  # 手动保存


# # 5. 执行解释和可视化
# # count=0
# # for data in test_loader:  # 假设这是您的测试数据
# #     if count >3:
# #         break
# #     data=data.to(device)
# #     print(data.y)
# #     print(model(data))
# #     explanation = explainer(
# #         x=data.x,
# #         edge_index=data.edge_index,
# #         edge_weight=getattr(data, 'edge_weight', None),
# #         batch=getattr(data, 'batch', None)
# #     )

# #     # origin_graph
# #     visualize_graph(data,count)
# #     # 可视化
# #     node_mask = explanation.node_mask.squeeze(-1)  # [num_nodes]
# #     edge_mask = explanation.edge_mask.squeeze()  # 关键修正：去除多余维度
# #     # explain_nodes(data, node_mask, edge_mask)
# #     # explain_subgraph(data, node_mask, edge_mask)

# #     # graph_explanation
# #     # 确保edge_mask是一维的
# #     #edge_mask = explanation.edge_mask.squeeze()  # 关键修正：去除多余维度
# #     # visualize_edge_explanation(data,edge_mask,count)
# #     node_edge_explanation(data, node_mask, edge_mask,count)
# #     #plt.close()
# #     count=count+1




# # #####图的可解释性与可视化
# def visualize_graph_embeddings(model, dataloader,layer):
#     """正确可视化图级嵌入（global_mean_pool后的结果）"""
#     model.eval()
#     all_embeddings = []
#     all_labels = []
    
#     with torch.no_grad():
#         for data in dataloader:
#             data = data.to(device)
#             _ = model(data)  # 前向传播
            
#             # 获取图级特征（假设通过global_mean_pool）
#             graph_emb = model.features[layer].mean(dim=0).unsqueeze(0)  # 全局平均
#             all_embeddings.append(graph_emb.cpu())
#             all_labels.append(data.y.cpu())
    
#     # 合并所有图的特征
#     embeddings = torch.cat(all_embeddings, dim=0).numpy()
#     labels = torch.cat(all_labels, dim=0).numpy()
    
    
#     # t-SNE降维
#     tsne = TSNE(n_components=2, random_state=42)
#     emb_2d = tsne.fit_transform(embeddings)
    
#     # 可视化
#     fig_width_cm = 17  # 全版图宽度
#     fig_height_cm = 15  # 高度
#     fig_width_inch = fig_width_cm * 0.3937
#     fig_height_inch = fig_height_cm * 0.3937
#     plt.figure(figsize=(fig_width_inch, fig_height_inch))
#     plt.scatter(
#     emb_2d[:, 0], 
#     emb_2d[:, 1], 
#     c=labels,                # 二值标签 [0, 1] 或 [1, 2]
#     cmap='bwr',              # 蓝-白-红渐变，适合二分类
#     s=50,                    # 点大小
#     edgecolors='k',          # 黑色边缘（增强对比）
#     alpha=0.8,               # 80%透明度（避免重叠遮挡）
#     linewidths=0.5           # 边缘线宽
# )

#     # 添加图例（显式标注类别）
#     import matplotlib.patches as mpatches
#     legend_patches = [
#         mpatches.Patch(color='blue', label='Class 0'),
#         mpatches.Patch(color='red', label='Class 1')
#     ]
#     plt.legend(handles=legend_patches,prop={'size': 10, 'family': 'Times New Roman'})
#     font = {'family' : 'Times New Roman',
#             'size'   : 14,
#             }
#     plt.title('t-SNE Visualization',fontdict=font)
#     # plt.rcParams['xtick.labelsize'] =8
#     # plt.rcParams['ytick.labelsize'] =8
#     plt.rcParams['font.family'] = 'Arial'  # 设置全局字体族（可选）
#     plt.rcParams['font.size'] = 8  
#     cb=plt.colorbar(fraction=0.046, pad=0.01, shrink=1)  # 颜色条标注 label='Class Probability', ticks=[0, 1]
#     cb.set_label(label='Class Probability',fontdict=font)
#     cb.set_ticks([0, 1])
#     plt.show()
#     plt.savefig(f'Tri_llama_{layer}.pdf', bbox_inches='tight')  # 手动保存
    


# # 可视化测试集所有图的嵌入分布,T-SNE
# # visualize_graph_embeddings(model,test_loader,'raw')
# # visualize_graph_embeddings(model,test_loader,'conv1')
# # visualize_graph_embeddings(model,test_loader,'global_mean_pool')