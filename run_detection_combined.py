import argparse
import os
#from six.moves import cPickle as pkl   #six提供python2和python3之间的兼容性
import pickle as pkl
from common_utils import *

torch.cuda.empty_cache()  # 清理未使用的显存
     
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama",choices=["llama",'llama-3','GLM',"Qwen",'mistral','DeepSeek'])
parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--dataset", default="halueval",choices=["halueval","ragtruth",'TruthfulQA', "fava", "fava_annot", "selfcheck"])
parser.add_argument(
    "--use_toklens", default=False ,action="store_true", help="remove prompt prefix before computing hidden and eigen scores"
)
parser.add_argument(
    "--mt", default=["logit", "hidden", "attns", 'Rel'],choices=["logit", "hidden", "attns", 'Rel'], action="append", help="choose method types for detection scores"
)


args = parser.parse_args()

if __name__ == "__main__":
    n_samples = args.n_samples
    model_name_or_path = get_full_model_name(args.model)[1]
    print(model_name_or_path)
    mt_list = args.mt

    print(
        f"Model: {args.model}, Method types: {mt_list}, Dataset: {args.dataset}, Use toklens: {args.use_toklens}"
    )

    # load dataset specific utils
    get_data, get_scores_dict = load_dataset_utils(args)  #get_data是一个函数

    train_sample_data, test_sample_data = get_data(n_samples=n_samples)  #sample_data, _分别为train和test

    # get scores for train sample data
    scores, sample_indiv_scores, sample_labels,hidden_acts,att_acts = get_scores_dict(model_name_or_path, train_sample_data, mt_list, args)
  
    file = f'InternalStates/LLM_Check_Hallucination_Detection-main/data/{args.dataset}/{args.model}/origin/scores_{args.dataset}_{args.model}_{n_samples}samp_train.pkl'
    try:
        with open(file, "wb") as f:
            pkl.dump([scores, sample_indiv_scores, sample_labels], f, protocol=pkl.HIGHEST_PROTOCOL)
        print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

    file1 = f'InternalStates/LLM_Check_Hallucination_Detection-main/data/{args.dataset}/{args.model}/origin/hidden_acts_labels_train_{args.dataset}_{args.model}1.pkl'
    try:
        with open(file1, "wb") as f:
            pkl.dump([hidden_acts, att_acts, sample_labels], f, protocol=pkl.HIGHEST_PROTOCOL)
        print("数据成功写入文件。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

    
    # get scores for test sample data
    # test_scores, test_sample_indiv_scores, test_sample_labels,test_hidden_acts,test_att_acts = get_scores_dict(model_name_or_path, test_sample_data, mt_list, args)

    # # save the scores to /data

    # # # 构建文件路径
    

    # file2 = f'InternalStates/LLM_Check_Hallucination_Detection-main/data/{args.dataset}/{args.model}/origin/scores_{args.dataset}_{args.model}_samp_test.pkl'
    # try:
    #     with open(file2, "wb") as f:
    #         pkl.dump([test_scores, test_sample_indiv_scores, test_sample_labels], f, protocol=pkl.HIGHEST_PROTOCOL)
    #     print("数据成功写入文件。")
    # except Exception as e:
    #     print(f"写入文件时发生错误：{e}")

    # file3 = f'InternalStates/LLM_Check_Hallucination_Detection-main/data/{args.dataset}/{args.model}/origin/hidden_acts_test_{args.dataset}_{args.model}.pkl'
    # try:
    #     with open(file3, "wb") as f:
    #         pkl.dump([test_hidden_acts, test_att_acts, test_sample_labels], f, protocol=pkl.HIGHEST_PROTOCOL)
    #     print("数据成功写入文件。")
    # except Exception as e:
    #     print(f"写入文件时发生错误：{e}")