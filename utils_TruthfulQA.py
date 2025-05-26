from collections import defaultdict
import pandas as pd
import jsonlines
import torch
from fastchat.model import get_conversation_template
from tqdm import tqdm


from common_utils import *


def get_truthful_data(n_samples=200):  #只选择了llama-2-7b-chat生成的回复，train_data大于200时结束
    train_data = []
    test_data = []
    trian_path='/data2/kklg/InternalStates/LLM_Check_Hallucination_Detection-main/truthfulQA/dataset/train.csv'
    test_path='/data2/kklg/InternalStates/LLM_Check_Hallucination_Detection-main/truthfulQA/dataset/test.csv'
    
    # 读取CSV文件
    df = pd.read_csv(trian_path, encoding='utf-8')
    # 遍历每一行数据
    for index, row in df.iterrows():
        # 处理每一行数据
        train_data.append(row)
    df = pd.read_csv(test_path, encoding='utf-8')
    for index, row in df.iterrows():
        # 处理每一行数据
        test_data.append(row)
    
    #print(len(train_data),len(test_data))  100 690
    return train_data, test_data   #train_data=test_data


def get_scores_dict(model_name_or_path, data, mt_list, args):
    system_prompt = ""
    generation_config = {}
    generation_config.update({"temperature": 0.6, "top_p": 0.9, "top_k": 50, "do_sample": True})
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, dtype=torch.bfloat16, **generation_config)
    tok_lens, labels, tok_ins = [], [], []

    scores = []
    indiv_scores = []
    # for mt in mt_list:
    #     indiv_scores[mt] = defaultdict(def_dict_value)
    #hidden_acts,att_acts保存最后一层的hidden states和对应的attention maps
    hidden_acts=[]
    att_acts=[]
    for i in tqdm(range(len(data))):  #len(data))
        # define the prompt, response and labels as per the dataset
        temp=[3,4]
        prompt = data[i][2]  #question
        choice=temp[i%2]
        response = data[i][choice]
        label = 1 if i%2 == 1 else 0   #0为真实，1为幻觉
        labels.append(label)

        chat_template = get_conversation_template(model_name_or_path)
        chat_template.messages=[]  #清空模板中原有的信息，重要！
        chat_template.set_system_message(system_prompt.strip())
        chat_template.append_message(chat_template.roles[0], prompt.strip())
        chat_template.append_message(chat_template.roles[1], response.strip())

        full_prompt = chat_template.get_prompt()
        user_prompt = full_prompt.split(response.strip())[0].strip()

        tok_in_u = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=True).input_ids
        tok_in = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).input_ids
        tok_lens.append([tok_in_u.shape[1], tok_in.shape[1]])

        logit, hidden_act, attn = get_model_vals(model, tok_in.to(0))
        # Unpacking the values into lists on CPU
        logit = logit[0].cpu()
        hidden_act = [x[0].to(torch.float32).detach().cpu() for x in hidden_act]
        attn = [x[0].to(torch.float32).detach().cpu() for x in attn]
        tok_in = tok_in.cpu()
        print(len(logit),len(hidden_act),len(attn))   #输出token数，隐层数，attns数
        tok_len = [tok_in_u.shape[1], tok_in.shape[1]]
        compute_scores(
            [logit],
            [hidden_act],
            [attn],
            scores,
            indiv_scores,
            mt_list,
            [tok_in],
            [tok_len],
            use_toklens=args.use_toklens,
        )
        hidden_acts.append(hidden_act[31])  #一共存储了33层的，每一层应该是m*d，保存每个样本的31层  中间层15
        att_acts.append(attn[31]) 
        print(scores, indiv_scores)
        print(indiv_scores)
    return scores, indiv_scores, labels ,hidden_acts,att_acts



