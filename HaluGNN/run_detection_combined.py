import argparse
import os
import pickle as pkl
from common_utils import *

torch.cuda.empty_cache()  # 
     
parser = argparse.ArgumentParser()   
parser.add_argument("--model", type=str, default="Qwen",choices=["llama",'llama-3',"Qwen"])
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--dataset", default="TriviaQA",choices=["ragtruth",'TruthfulQA', "sciQ", "TriviaQA"])

args = parser.parse_args()

if __name__ == "__main__":
    n_samples = args.n_samples
    model_name_or_path = get_full_model_name(args.model)[1]
    print(model_name_or_path)

    print(
        f"Model: {args.model}, Dataset: {args.dataset}, Use toklens: {args.use_toklens}"
    )

    # load dataset specific utils
    get_data, get_scores_dict = load_dataset_utils(args)  #get_data是一个函数

    train_sample_data, test_sample_data = get_data(n_samples=n_samples)  #sample_data, _分别为train和test

    # get scores for train sample data
    scores, sample_indiv_scores, sample_labels,hidden_acts,att_acts = get_scores_dict(model_name_or_path, train_sample_data, mt_list, args)
  
    file = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/origin/scores_{args.dataset}_{args.model}_{n_samples}samp_train.pkl'
    try:
        with open(file, "wb") as f:
            pkl.dump([scores, sample_indiv_scores, sample_labels], f, protocol=pkl.HIGHEST_PROTOCOL)
        print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

    file1 = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/origin/hidden_acts_labels_train_{args.dataset}_{args.model}.pkl'
    try:
        with open(file1, "wb") as f:
            pkl.dump([hidden_acts, att_acts, sample_labels], f, protocol=pkl.HIGHEST_PROTOCOL)
        print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

    
    # get scores for test sample data
    test_scores, test_sample_indiv_scores, test_sample_labels,test_hidden_acts,test_att_acts = get_scores_dict(model_name_or_path, test_sample_data, mt_list, args)

    # save the scores to /data

    # # 构建文件路径
    

    file2 = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/origin/scores_{args.dataset}_{args.model}_samp_test.pkl'
    try:
        with open(file2, "wb") as f:
            pkl.dump([test_scores, test_sample_indiv_scores, test_sample_labels], f, protocol=pkl.HIGHEST_PROTOCOL)
        print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

    file3 = f'/HaluGNN/internalstates_data/{args.dataset}/{args.model}/origin/hidden_acts_test_{args.dataset}_{args.model}.pkl'
    try:
        with open(file3, "wb") as f:
            pkl.dump([test_hidden_acts, test_att_acts, test_sample_labels], f, protocol=pkl.HIGHEST_PROTOCOL)
        print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")