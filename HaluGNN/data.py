#####生成训练和测试数据，并保存
import torch
import argparse
from torch_geometric.data import Data
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",  default='sciQ',choices=['ragtruth','halueval','TruthfulQA', "sciQ", "TriviaQA"])
parser.add_argument("--model", default='llama',choices=["llama",'llama-3','GLM',"Qwen",'mistral','DeepSeek'])
parser.add_argument("--type", default='screen',choices=["screen",'raw'])

args = parser.parse_args()

#原始隐层状态的文件路径
train_file_name = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/origin/hidden_acts_labels_train_{args.dataset}_{args.model}.pkl'
test_file_name = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/origin/hidden_acts_test_{args.dataset}_{args.model}.pkl'

#保存将隐层状态转换为图的信息
train_data_base =f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/{args.type}/train_data_{args.dataset}_{args.model}.pkl'
test_data_base = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/{args.type}/test_data_{args.dataset}_{args.model}.pkl'

#将图信息转换为Data格式
train_data_Data = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/{args.type}/{args.dataset}_{args.model}_train_data_list.pkl'
test_data_Data = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/{args.type}/{args.dataset}_{args.model}_test_data_list.pkl'

def create_data_list(token_hidden_states, edge_index_list,edge_weight_list,labels):
    data_list = []
    for hidden_states, edge_index,edge_weight,label in zip(token_hidden_states, edge_index_list,edge_weight_list,labels):
        x = hidden_states.clone().detach()
        y = torch.tensor([label], dtype=torch.long)
        edge_index = edge_index  #torch.tensor([[i, i+1] for i in range(len(x)-1)], dtype=torch.long).t().contiguous()
        edge_weight = edge_weight
        data = Data(x=x, edge_index=edge_index, edge_weight = edge_weight,y=y)
        data_list.append(data)
    return data_list
#train_file_name = f'/data2/kklg/internalstates_data/ceshi/hidden_acts_labels.pkl'
def generate_index_weight():
    # 使用with语句确保文件正确关闭
    with open(train_file_name, 'rb') as file1:
        # 加载pkl文件内容
        train_data = pkl.load(file1)

    # 使用with语句确保文件正确关闭
    with open(test_file_name, 'rb') as file2:
        # 加载pkl文件内容
        test_data = pkl.load(file2)

    trian_hidden_states_list, train_edge_list, train_y_list=train_data[0],train_data[1],train_data[2]
    test_hidden_states_list, test_edge_list, test_y_list=test_data[0],test_data[1],test_data[2]

    train_edge_index_list=[]
    train_edge_weight_list=[]
    # 遍历下三角矩阵
    for edge in train_edge_list:
        edge_index = []
        edge_weight = []
        edge = edge[0]
        for i in range(edge.size(0)):
            for j in range(i+1):
                # if i ==j :
                #     break
                if edge[i, j] > 0.2:  #对边的权重进行筛选
                    edge_index.append([j, i])  #边的起点和终点
                    edge_weight.append(edge[i, j])
        # 转换为张量
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weight = torch.tensor(edge_weight)
        
        #print(edge_index )
        #print(edge_weight)
        train_edge_index_list.append(edge_index)
        train_edge_weight_list.append(edge_weight)

    test_edge_index_list=[]
    test_edge_weight_list=[]
    # 遍历下三角矩阵
    for edge in test_edge_list:
        edge_index = []
        edge_weight = []
        edge = edge[0]
        for i in range(edge.size(0)):
            for j in range(i+1):
                # if i ==j :
                #     break
                if edge[i, j] > 0.2:  #对边的权重进行筛选
                    edge_index.append([j, i])  #边的起点和终点
                    edge_weight.append(edge[i, j])
        # 转换为张量
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weight = torch.tensor(edge_weight)
        test_edge_index_list.append(edge_index)
        test_edge_weight_list.append(edge_weight)
    
    #保存原始训练集和测试集
    train_data = train_data_base
    try:
        with open(train_data, "wb") as f:
            pkl.dump([trian_hidden_states_list, train_edge_index_list,train_edge_weight_list,train_y_list], f, protocol=pkl.HIGHEST_PROTOCOL)
            print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")


    test_data = test_data_base
    try:
        with open(test_data, "wb") as f:
            pkl.dump([test_hidden_states_list, test_edge_index_list,test_edge_weight_list,test_y_list], f, protocol=pkl.HIGHEST_PROTOCOL)
            print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

def generate_Data():
    # 使用with语句确保文件正确关闭
    with open(train_data_base, 'rb') as file1:
        # 加载pkl文件内容
        train_data = pkl.load(file1)

    #使用with语句确保文件正确关闭
    with open(test_data_base, 'rb') as file2:
        # 加载pkl文件内容
        test_data = pkl.load(file2)

    train_data_list = create_data_list(train_data[0],train_data[1],train_data[2],train_data[3])
    #print(train_data_list)
    test_data_list = create_data_list(test_data[0],test_data[1],test_data[2],test_data[3])

    #保存训练集和测试集
    
    try:
        with open(train_data_Data, "wb") as f:
            pkl.dump(train_data_list, f, protocol=pkl.HIGHEST_PROTOCOL)
            print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")


    
    try:
        with open(test_data_Data, "wb") as f:
            pkl.dump(test_data_list, f, protocol=pkl.HIGHEST_PROTOCOL)
            print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

# generate_index_weight()
# generate_Data()
# with open(train_data_base, 'rb') as file1:
#     # 加载pkl文件内容
#     train_data = pkl.load(file1)
# for hidden_states, edge_index,edge_weight,label in zip(train_data[0],train_data[1],train_data[2],train_data[3]):
#     x = hidden_states.clone().detach()
#     print(x.shape)