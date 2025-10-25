
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import util
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#dataset
def load_dataset_utils(args):
    """Util to lead different datasets"""

    if args.dataset == "sciQ":  #Available
        import utils_SciQ

        get_data = utils_SciQ.get_SciQ_data
        get_scores_dict = utils_SciQ.get_scores_dict
    elif args.dataset == "ragtruth":  #Available
        import utils_ragtruth

        get_data = utils_ragtruth.get_ragtruth_data
        get_scores_dict = utils_ragtruth.get_scores_dict
    elif args.dataset == "TruthfulQA": #Available
        import utils_TruthfulQA

        get_data = utils_TruthfulQA.get_truthful_data
        get_scores_dict = utils_TruthfulQA.get_scores_dict
    elif args.dataset == "TriviaQA":  #Available
        import utils_TriviaQA

        get_data = utils_TriviaQA.get_TriviaQA_data
        get_scores_dict = utils_TriviaQA.get_scores_dict
    else:
        raise ValueError("Invalid dataset")
    return get_data, get_scores_dict



#model name and path
def get_full_model_name(model_name: str):
    """Map a short model name identifier to a fully qualified model name"""
    if "llama-3" in model_name:  #Available
        name = ["llama-3", "/data/Meta-Llama-3-8B-Instruct"]
    elif "llama" in model_name:  #Available
        name = ["llama", "/data2/kklg/Llama2-7B-chat-hf"] 
    elif "Qwen" in model_name:  #Available
        name = ["Qwen", "/data/Qwen2-7B-Instruct"]
    return name

#load model
def load_model_and_tokenizer(
    model_name_or_path: str, dtype=torch.float32, **kwargs
):
    """Util to load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True,device_map='auto',torch_dtype=torch.float16,**kwargs)
    #model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16,**kwargs).to(device)
    model.requires_grad_(False)
    if model.generation_config.temperature is None:
        model.generation_config.temperature = 1.0
    model.generation_config.do_sample = True

    #tokenizer_name_or_path = model_name_or_path
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"

    return model, tokenizer
