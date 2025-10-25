from collections import defaultdict
import pandas as pd
import jsonlines
import torch
from fastchat.model import get_conversation_template
from tqdm import tqdm


from common_utils import *


def get_SciQ_data(n_samples=200):  
    train_data = []
    test_data = []
    path='/HaluGNN/SciQ/dataset/'
    data = pd.read_parquet(path)
    for index, row in data.iterrows():
        if  len(train_data) < n_samples:
            train_data.append(row)
        elif len(train_data) >= n_samples and len(test_data) < 1000-n_samples:
            test_data.append(row)
    return train_data, test_data  


def get_scores_dict(model_name_or_path, data, args):
    system_prompt = ""
    generation_config = {}
    generation_config.update({"temperature": 0.6, "top_p": 0.9, "top_k": 50, "do_sample": True})
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, dtype=torch.float16, **generation_config)
    tok_lens, labels, tok_ins = [], [], []

    hidden_acts=[]
    att_acts=[]
    for i in tqdm(range(len(data))):  #len(data))
        # define the prompt, response and labels as per the dataset
        temp=['correct_answer','distractor1']
        prompt = data[i]["question"]
        choice=temp[i%2]
        response = data[i][choice]
        label = 1 if i%2 == 1 else 0   
        labels.append(label)

        chat_template = get_conversation_template(model_name_or_path)
        chat_template.messages=[]  #
        chat_template.set_system_message(system_prompt.strip())
        chat_template.append_message(chat_template.roles[0], prompt.strip())
        chat_template.append_message(chat_template.roles[1], response.strip())

        full_prompt = chat_template.get_prompt()
        user_prompt = full_prompt.split(response.strip())[0].strip()
        #print(full_prompt)
        #print(user_prompt)
        
        tok_in_u = tokenizer(user_prompt, return_tensors="pt", add_special_tokens=True).input_ids
        tok_in = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True).input_ids
        #print("Attention Mask:", tok_in
        # print("Input IDs:", inputs["input_ids"])
        # print("Attention Mask:", inputs["attention_mask"])
        tok_lens.append([tok_in_u.shape[1], tok_in.shape[1]])

        logit, hidden_act, attn = get_model_vals(model, tok_in)
        # Unpacking the values into lists on CPU
        logit = logit[0].cpu()
        hidden_act = [x[0].to(torch.float32).detach().cpu() for x in hidden_act]
        attn = [x[0].to(torch.float32).detach().cpu() for x in attn]
       
        hidden_acts.append(hidden_act[-2])  #
        att_acts.append(attn[-1]) 
    return labels ,hidden_acts,att_acts



