# import numpy as np
# import torch

# # 假设Sigma是一个已经给定的下三角方阵
# Sigma = torch.tensor([[1,0,0],[1,2,0],[1,2,3]])  # 这里应该填入具体的Sigma矩阵数据
# i1 = Sigma.shape[0]  # Sigma方阵的大小

# # 提取对角线元素作为特征值
# eigenvalues = torch.diagonal(Sigma)

# # 计算需要选择的特征值数量（前2/3个）
# num_select = int(np.ceil((2/3) * i1))

# # 找到前2/3个最大的特征值的下标
# indices = np.argsort(eigenvalues)[-num_select:]
# print(indices)
# # 假设tok_in是一个已经给定的数组或列表
# tok_in = [4,5,6]  # 这里应该填入具体的tok_in数据

# # 将特征值下标对应到tok_in的位置
# selected_tok_in = [tok_in[idx] for idx in indices]

# print("选中的tok_in位置:", selected_tok_in)

'''

生成ragtruth文件
import json

# 指定输入文件列表
input_file1 = '/data2/kklg/InternalStates/LLM_Check_Hallucination_Detection-main/rag_truth/dataset/source_info.jsonl'
input_file2 = '/data2/kklg/InternalStates/LLM_Check_Hallucination_Detection-main/rag_truth/dataset/response.jsonl'

# 指定输出文件
output_file = '/data2/kklg/InternalStates/LLM_Check_Hallucination_Detection-main/rag_truth/dataset/ragtruth.jsonl'

# 定义要提取的元素列表
keys_to_extract_file1 = ["prompt"]
keys_to_extract_file2 = ["model", "response","split",'labels']

# 提取元素并返回新字典
def extract_elements(data, keys_to_extract):
    return {key: data[key] for key in keys_to_extract if key in data}

# 处理文件并写入新文件
def process_files_and_write(input_file1, input_file2, output_file, keys_to_extract_file1, keys_to_extract_file2):
    with open(input_file1, 'r', encoding='utf-8') as file1, \
         open(input_file2, 'r', encoding='utf-8') as file2, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line1 in file1:
            data1 = json.loads(line1)
            if data1["task_type"]=='Summary':
                extracted_data1 = extract_elements(data1, keys_to_extract_file1)
                for line2 in file2:
                    data2 = json.loads(line2)
                    if data2["model"]=="llama-2-7b-chat"  and data2['source_id']==data1['source_id']:
                        extracted_data2 = extract_elements(data2, keys_to_extract_file2)
                        break
                # 合并两个字典
                merged_data = {**extracted_data1, **extracted_data2}
                
                # 写入新文件
                json.dump(merged_data, outfile)
                outfile.write('\n')
                    
                
            
        # for line2 in file2:
        #     data2 = json.loads(line2)

        #     if data2["model"]=="llama-2-7b-chat":
        #         extracted_data2 = extract_elements(data2, keys_to_extract_file2)
            
                
    

        # for line1, line2 in zip(file1, file2):  #line表示一行内容
            
        #     data1 = json.loads(line1)
        #     data2 = json.loads(line2)

        #     if data1["task_type"]=='Summary'  and data2["model"]=="llama-2-7b-chat" :
        #         extracted_data1 = extract_elements(data1, keys_to_extract_file1)
        #         extracted_data2 = extract_elements(data2, keys_to_extract_file2)
            
        #         # 合并两个字典
        #         merged_data = {**extracted_data1, **extracted_data2}
                
        #         # 写入新文件
        #         json.dump(merged_data, outfile)
        #         outfile.write('\n')



# 执行函数
process_files_and_write(input_file1, input_file2, output_file, keys_to_extract_file1, keys_to_extract_file2)
'''


# 确保文件所在的目录存在
import pickle
import os

# # 假设 file 是一个已经定义好的文件路径字符串
# file=f"InternalStates/LLM_Check_Hallucination_Detection-main/data/scores.pkl"

# # 确保文件所在的目录存在
# os.makedirs(os.path.dirname(file), exist_ok=True)
# array=[
# [[1.781470775604248, 7.394536805804819e-05, 0.0045972797, -1.0644938945770264, 0.6121405363082886, 1.63495934009552, 2.3088161945343018, 2.8927643299102783, 3.3641927242279053, 3.690626859664917, 3.955063819885254, 4.246875762939453, 4.477787494659424, 4.662363529205322, 4.854955673217773, 5.03356409072876, 5.227082252502441, 5.4329328536987305, 5.678644180297852, 5.944613933563232, 6.14424467086792, 6.413542747497559, 6.653402328491211, 6.8596906661987305, 7.044315814971924, 7.235429763793945, 7.403791904449463, 7.554837226867676, 7.699976444244385, 7.84686803817749, 7.983938217163086, 8.114632606506348, 8.227385520935059, 8.359577178955078, 8.845598220825195, -141.4866943359375, -164.03878784179688, -176.9967498779297, -171.38587951660156, -180.07257080078125, -189.2906951904297, -182.50765991210938, -183.05490112304688, -159.74696350097656, -167.1936798095703, -181.3799285888672, -154.9128875732422, -156.5019073486328, -153.22052001953125, -145.46315002441406, -156.6012420654297, -157.56146240234375, -155.07577514648438, -167.63394165039062, -171.34432983398438, -190.03475952148438, -169.40130615234375, -165.1348114013672, -180.65762329101562, -182.4456024169922, -166.46926879882812, -154.6486053466797, -163.07720947265625, -166.90762329101562, -163.3819580078125, -149.16265869140625, 0.8771196603775024, 0.060848258435726166, 0.09394755959510803, 0.09805917739868164, 0.14366228878498077, 0.10695002973079681, 0.10742215812206268, 0.1110941618680954, 0.08180983364582062, 0.10344291478395462, 0.09454500675201416, 0.10327678918838501, 0.10385368764400482, 0.08690442889928818, 0.1240762397646904, 0.1610105335712433, 0.17699821293354034, 0.18537119030952454, 0.17503270506858826, 0.1820465326309204, 0.23302897810935974, 0.23549550771713257, 0.24867209792137146, 0.26514992117881775, 0.3328293263912201, 0.2792813777923584, 0.3265075981616974, 0.3315007984638214, 0.3436621427536011, 0.3856792747974396, 0.36309266090393066]], 
# {
# 'logit':  {'perplexity': [1.781470775604248], 'window_entropy': [7.394536805804819e-05], 'logit_entropy': [0.0045972797]}, 'hidden': {'Hly1': [-1.0644938945770264], 'Hly2': [0.6121405363082886], 'Hly3': [1.63495934009552], 'Hly4': [2.3088161945343018], 'Hly5': [2.8927643299102783], 'Hly6': [3.3641927242279053], 'Hly7': [3.690626859664917], 'Hly8': [3.955063819885254], 'Hly9': [4.246875762939453], 'Hly10': [4.477787494659424], 'Hly11': [4.662363529205322], 'Hly12': [4.854955673217773], 'Hly13': [5.03356409072876], 'Hly14': [5.227082252502441], 'Hly15': [5.4329328536987305], 'Hly16': [5.678644180297852], 'Hly17': [5.944613933563232], 'Hly18': [6.14424467086792], 'Hly19': [6.413542747497559], 'Hly20': [6.653402328491211], 'Hly21': [6.8596906661987305], 'Hly22': [7.044315814971924], 'Hly23': [7.235429763793945], 'Hly24': [7.403791904449463], 'Hly25': [7.554837226867676], 'Hly26': [7.699976444244385], 'Hly27': [7.84686803817749], 'Hly28': [7.983938217163086], 'Hly29': [8.114632606506348], 'Hly30': [8.227385520935059], 'Hly31': [8.359577178955078], 'Hly32': [8.845598220825195]}, 'attns': {'Attn1': [-141.4866943359375], 'Attn2': [-164.03878784179688], 'Attn3': [-176.9967498779297], 'Attn4': [-171.38587951660156], 'Attn5': [-180.07257080078125], 'Attn6': [-189.2906951904297], 'Attn7': [-182.50765991210938], 'Attn8': [-183.05490112304688], 'Attn9': [-159.74696350097656], 'Attn10': [-167.1936798095703], 'Attn11': [-181.3799285888672], 'Attn12': [-154.9128875732422], 'Attn13': [-156.5019073486328], 'Attn14': [-153.22052001953125], 'Attn15': [-145.46315002441406], 'Attn16': [-156.6012420654297], 'Attn17': [-157.56146240234375], 'Attn18': [-155.07577514648438], 'Attn19': [-167.63394165039062], 'Attn20': [-171.34432983398438], 'Attn21': [-190.03475952148438], 'Attn22': [-169.40130615234375], 'Attn23': [-165.1348114013672], 'Attn24': [-180.65762329101562], 'Attn25': [-182.4456024169922], 'Attn26': [-166.46926879882812], 'Attn27': [-154.6486053466797], 'Attn28': [-163.07720947265625], 'Attn29': [-166.90762329101562], 'Attn30': [-163.3819580078125], 'Attn31': [-149.16265869140625]}, 'Rel':  {'Rel1': [0.8771196603775024], 'Rel2': [0.060848258435726166], 'Rel3': [0.09394755959510803], 'Rel4': [0.09805917739868164], 'Rel5': [0.14366228878498077], 'Rel6': [0.10695002973079681], 'Rel7': [0.10742215812206268], 'Rel8': [0.1110941618680954], 'Rel9': [0.08180983364582062], 'Rel10': [0.10344291478395462], 'Rel11': [0.09454500675201416], 'Rel12': [0.10327678918838501], 'Rel13': [0.10385368764400482], 'Rel14': [0.08690442889928818], 'Rel15': [0.1240762397646904], 'Rel16': [0.1610105335712433], 'Rel17': [0.17699821293354034], 'Rel18': [0.18537119030952454], 'Rel19': [0.17503270506858826], 'Rel20': [0.1820465326309204], 'Rel21': [0.23302897810935974], 'Rel22': [0.23549550771713257], 'Rel23': [0.24867209792137146], 'Rel24': [0.26514992117881775], 'Rel25': [0.3328293263912201], 'Rel26': [0.2792813777923584], 'Rel27': [0.3265075981616974], 'Rel28': [0.3315007984638214], 'Rel29': [0.3436621427536011], 'Rel30': [0.3856792747974396], 'Rel31': [0.36309266090393066]}
# }, 

# [1]]

# dict=dict({
# 'logit': defaultdict(<function def_dict_value at 0x792a74f5c160>, {'perplexity': [1.781470775604248], 'window_entropy': [7.394536805804819e-05], 'logit_entropy': [0.0045972797]}), 
# 'hidden': defaultdict(<function def_dict_value at 0x792a74f5c160>, {'Hly1': [-1.0644938945770264], 'Hly2': [0.6121405363082886], 'Hly3': [1.63495934009552], 'Hly4': [2.3088161945343018], 'Hly5': [2.8927643299102783], 'Hly6': [3.3641927242279053], 'Hly7': [3.690626859664917], 'Hly8': [3.955063819885254], 'Hly9': [4.246875762939453], 'Hly10': [4.477787494659424], 'Hly11': [4.662363529205322], 'Hly12': [4.854955673217773], 'Hly13': [5.03356409072876], 'Hly14': [5.227082252502441], 'Hly15': [5.4329328536987305], 'Hly16': [5.678644180297852], 'Hly17': [5.944613933563232], 'Hly18': [6.14424467086792], 'Hly19': [6.413542747497559], 'Hly20': [6.653402328491211], 'Hly21': [6.8596906661987305], 'Hly22': [7.044315814971924], 'Hly23': [7.235429763793945], 'Hly24': [7.403791904449463], 'Hly25': [7.554837226867676], 'Hly26': [7.699976444244385], 'Hly27': [7.84686803817749], 'Hly28': [7.983938217163086], 'Hly29': [8.114632606506348], 'Hly30': [8.227385520935059], 'Hly31': [8.359577178955078], 'Hly32': [8.845598220825195]}), 
# 'attns': defaultdict(<function def_dict_value at 0x792a74f5c160>, {'Attn1': [-141.4866943359375], 'Attn2': [-164.03878784179688], 'Attn3': [-176.9967498779297], 'Attn4': [-171.38587951660156], 'Attn5': [-180.07257080078125], 'Attn6': [-189.2906951904297], 'Attn7': [-182.50765991210938], 'Attn8': [-183.05490112304688], 'Attn9': [-159.74696350097656], 'Attn10': [-167.1936798095703], 'Attn11': [-181.3799285888672], 'Attn12': [-154.9128875732422], 'Attn13': [-156.5019073486328], 'Attn14': [-153.22052001953125], 'Attn15': [-145.46315002441406], 'Attn16': [-156.6012420654297], 'Attn17': [-157.56146240234375], 'Attn18': [-155.07577514648438], 'Attn19': [-167.63394165039062], 'Attn20': [-171.34432983398438], 'Attn21': [-190.03475952148438], 'Attn22': [-169.40130615234375], 'Attn23': [-165.1348114013672], 'Attn24': [-180.65762329101562], 'Attn25': [-182.4456024169922], 'Attn26': [-166.46926879882812], 'Attn27': [-154.6486053466797], 'Attn28': [-163.07720947265625], 'Attn29': [-166.90762329101562], 'Attn30': [-163.3819580078125], 'Attn31': [-149.16265869140625]}), 
# 'Rel': defaultdict(<function def_dict_value at 0x792a74f5c160>, {'Rel1': [array([[0.87711966]], dtype=float32)], 'Rel2': [array([[0.06084826]], dtype=float32)], 'Rel3': [array([[0.09394756]], dtype=float32)], 'Rel4': [array([[0.09805918]], dtype=float32)], 'Rel5': [array([[0.14366229]], dtype=float32)], 'Rel6': [array([[0.10695003]], dtype=float32)], 'Rel7': [array([[0.10742216]], dtype=float32)], 'Rel8': [array([[0.11109416]], dtype=float32)], 'Rel9': [array([[0.08180983]], dtype=float32)], 'Rel10': [array([[0.10344291]], dtype=float32)], 'Rel11': [array([[0.09454501]], dtype=float32)], 'Rel12': [array([[0.10327679]], dtype=float32)], 'Rel13': [array([[0.10385369]], dtype=float32)], 'Rel14': [array([[0.08690443]], dtype=float32)], 'Rel15': [array([[0.12407624]], dtype=float32)], 'Rel16': [array([[0.16101053]], dtype=float32)], 'Rel17': [array([[0.17699821]], dtype=float32)], 'Rel18': [array([[0.18537119]], dtype=float32)], 'Rel19': [array([[0.1750327]], dtype=float32)], 'Rel20': [array([[0.18204653]], dtype=float32)], 'Rel21': [array([[0.23302898]], dtype=float32)], 'Rel22': [array([[0.23549551]], dtype=float32)], 'Rel23': [array([[0.2486721]], dtype=float32)], 'Rel24': [array([[0.26514992]], dtype=float32)], 'Rel25': [array([[0.33282933]], dtype=float32)], 'Rel26': [array([[0.27928138]], dtype=float32)], 'Rel27': [array([[0.3265076]], dtype=float32)], 'Rel28': [array([[0.3315008]], dtype=float32)], 'Rel29': [array([[0.34366214]], dtype=float32)], 'Rel30': [array([[0.38567927]], dtype=float32)], 'Rel31': [array([[0.36309266]], dtype=float32)]})
# })

# try:
#     with open(file, "wb") as f:
#         pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print("数据成功写入文件。")
# except Exception as e:
#     print(f"写入文件时发生错误：{e}")

# import pickle

# # 替换为您的pkl文件路径
# file_path = f'/data2/kklg/InternalStates/LLM_Check_Hallucination_Detection-main/data/halueval/llama/origin/hidden_acts_labels_train_halueval_llama1.pkl'

# # 读取pkl文件
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
#     trian_hidden_states_list, train_edge_list, train_y_list=data[0],data[1],data[2]

# # 打印读取的数据
# print(trian_hidden_states_list)
# print(train_edge_list)
# print(train_y_list)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/data2/kklg/Llama2-7B-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.float16)

# # 示例输入   没问题
# text = "Hello, how are you?"
# inputs = tokenizer(text, return_tensors="pt")

# # 查看 attention_mask
# print("Attention Mask:", inputs["attention_mask"])
# 正常情况应该是全1，如 tensor([[1, 1, 1, 1, 1, 1]])

# 测试极短输入（如长度=3）
inputs_short = tokenizer("Hi!", return_tensors="pt")
outputs_short = model(**inputs_short, output_attentions=True)

# 检查第17层注意力是否仍然为0
attentions = outputs_short.attentions  # 所有层的注意力矩阵
print("Layer16 Attention Sum:", attentions[16].sum())  # 第17层（索引16）
print("Layer31 Attention Sum:", attentions[31].sum() ) # 第32层（索引31）
