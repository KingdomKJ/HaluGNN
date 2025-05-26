#少了rel的部分误删了
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import util
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#返回数据集
def load_dataset_utils(args):
    """Util to lead different datasets"""
    if args.dataset == "selfcheck":
        import utils_selfcheck

        get_data = utils_selfcheck.get_selfcheck_data
        get_scores_dict = utils_selfcheck.get_scores_dict
    elif args.dataset == "fava":
        import utils_fava

        get_data = utils_fava.get_fava_data
        get_scores_dict = utils_fava.get_scores_dict
    elif args.dataset == "fava_annot":
        import utils_fava_annotated

        get_data = utils_fava_annotated.get_fava_data
        get_scores_dict = utils_fava_annotated.get_scores_dict
    elif args.dataset == "ragtruth":  #可用
        import utils_ragtruth

        get_data = utils_ragtruth.get_ragtruth_data
        get_scores_dict = utils_ragtruth.get_scores_dict
    elif args.dataset == "TruthfulQA": #可用
        import utils_TruthfulQA

        get_data = utils_TruthfulQA.get_truthful_data
        get_scores_dict = utils_TruthfulQA.get_scores_dict
    elif args.dataset == "halueval":  #可用
        import utils_halueval

        get_data = utils_halueval.get_halueval_data
        get_scores_dict = utils_halueval.get_scores_dict
    else:
        raise ValueError("Invalid dataset")
    return get_data, get_scores_dict

#评价指标ARC、ACC、TPR
def get_roc_scores(scores: np.array, labels: np.array):
    """
    Util to get area under the curve, accuracy and tpr at 5% fpr
    Args:
        scores (np.array): Scores for the prediction
        labels (np.array): Ground Truth Labels

    Returns:
        arc (float): area under the curve
        accuracy (float): accuracy at best TPR and FPR selection
        low (float): TPR at 5% FPR
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.05)[0][-1]]
    return arc, acc, low

#评价指标
def get_roc_auc_scores(scores: np.array, labels: np.array):
    """
    Util to get area under the curve, accuracy and tpr at 5% fpr
    Args:
        scores (np.array): Scores for the prediction
        labels (np.array): Ground Truth Labels

    Returns:
        arc (float): area under the curve
        accuracy (float): accuracy at best TPR and FPR selection
        low (float): TPR at 5% FPR
        fpr (np.array): Array with False Positive Values
        tpr (np.array): Array with True Positive Values
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.05)[0][-1]]
    return arc, acc, low, fpr, tpr

#模型名称——模型存放的路径
def get_full_model_name(model_name: str):
    """Map a short model name identifier to a fully qualified model name"""
    if "vicuna7b" in model_name:
        name = ["vicuna", "lmsys/vicuna-7b-v1.5"]
    elif "vicuna13b" in model_name:
        name = ["vicuna13b", "lmsys/vicuna-13b-v1.5"]
    elif "llama-3" in model_name:  #可用
        name = ["llama-3", "/data/Meta-Llama-3-8B-Instruct"]
    elif "llama" in model_name:  #可用
        name = ["llama", "/data2/kklg/Llama2-7B-chat-hf"]   #/data/Llama-2-7b  /data2/kklg/Llama2-7B-chat-hf
    elif "GLM" in model_name:  #可用
        name = ["GLMGLM", "/data/GLM-4-9b"]
    elif "mistral" in model_name:  #可用,但是为稀疏激活
        name = ["mistral", "/data/Mistral-7B-Instruct-v0.3"]
    elif "Qwen" in model_name:  #可用,但是为稀疏激活
        name = ["Qwen", "/data/Qwen2-7B-Instruct"]
    elif "DeepSeek" in model_name:  #可用,但是为稀疏激活
        name = ["DeepSeek", "/data/Deepseek-7b-chat"]
    elif "pythia" in model_name:
        name = ["pythia", "togethercomputer/Pythia-Chat-Base-7B"]
    elif "guanaco" in model_name:
        name = ["guanaco", "JosephusCheung/Guanaco"]
    elif "falcon" in model_name:
        name = ["falcon", "tiiuae/falcon-7b-instruct"]
    return name

#加载模型和分词器
def load_model_and_tokenizer(
    model_name_or_path: str, dtype=torch.float32, **kwargs
):
    """Util to load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",torch_dtype=torch.float16)
    #model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=dtype, **kwargs).to(device)
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

#计算文中提出的各种Scores
def compute_scores(logits, hidden_acts, attns, scores, indiv_scores, mt_list, tok_ins, tok_lens, use_toklens=True):
    """Compute various evaluation scores (e.g., perplexity, entropy, SVD scores) from model outputs.
    
    #模型输出logits,hidden states,attentions
    This function takes model outputs (logits, hidden states, attentions) and computes
    a list of metric scores defined by `mt_list`. The computed scores are appended
    to `scores` and `indiv_scores` dictionaries for tracking.

    NOTE: The indiv_scores score dictionary will be saved to disk and then used for final metric computation in
    check scores ipynb

    Args:
        logits: Model logits.
        hidden_acts: Hidden activations.
        attns: Attention matrices.
        scores (list): A list to store aggregated scores across samples.
        indiv_scores (dict): A dictionary to store metric-specific scores for each sample
        mt_list (list): A list of metric types to compute.
        tok_ins: A list of tokenized inputs for each sample.  #输入文本的tokens
        tok_lens: A list of tuples indicating the start and end token indices for each sample.  #每个样本的起止下标
        use_toklens (bool, optional): Whether to use `tok_lens` to slice sequences. Defaults to True.   #

    Raises:
        ValueError: If an invalid metric type is encountered in `mt_list`.
    """
    sample_scores = []
    hly=[]
    attn=[]
    rel=[]
    for mt in mt_list:
        mt_score = []  #每个指标一个列表
        
        if mt == "logit":
            mt_score.append(perplexity(logits, tok_ins, tok_lens)[0])
            per_score=mt_score[-1]
            #indiv_scores[mt]["perplexity"].append(mt_score[-1])

            mt_score.append(window_logit_entropy(logits, tok_lens, w=1)[0])
            win_en = mt_score[-1]
            #indiv_scores[mt]["window_entropy"].append(mt_score[-1])

            mt_score.append(logit_entropy(logits, tok_lens, top_k=50)[0])
            lo_en= mt_score[-1]
            #indiv_scores[mt]["logit_entropy"].append(mt_score[-1])

        elif mt == "hidden":
            
            for layer_num in range(0, len(hidden_acts[0])-1): 
                mt_score.append(get_svd_eval(hidden_acts, layer_num, tok_lens, use_toklens)[0])
                hly.append(mt_score[-1])
                #indiv_scores[mt]["Hly" + str(layer_num)].append(mt_score[-1])

        elif mt == "attns":
            
            for layer_num in range(0, len(attns[0])):
                mt_score.append(get_attn_eig_prod(attns, layer_num, tok_lens, use_toklens)[0])
                attn.append(mt_score[-1]) 
                #indiv_scores[mt]["Attn" + str(layer_num)].append(mt_score[-1])
        
        elif mt == "Rel":
            
            for layer_num in range(0, len(attns[0])):  #0到len(attns[0])-1
                mt_score.append(get_Rel_Prom_Res(hidden_acts, attns, layer_num, tok_lens, use_toklens)[0])
                rel.append(mt_score[-1])
                #indiv_scores[mt]["Rel" + str(layer_num)].append(mt_score[-1])

        else:
            raise ValueError("Invalid method type")
        
        log = {'logit': {
                "perplexity":per_score,
                "window_entropy":win_en ,
                "logit_entropy":lo_en,
               },
                'hidden': hly,  
                'attns': attn, 
                'Rel': rel,  
        }

        sample_scores.append(mt_score)  #添加每个样本的对应指标
        

    scores.append(sample_scores)   #添加所有样本的指标
    #print(scores)
    indiv_scores.append(log)  #添加每个样本的log


def get_model_vals(model, tok_in):
    """Run the model forward pass to obtain logits, hidden states, and attention scores.

    Args:
        model: A pretrained model compatible with the transformers API.
        tok_in (torch.Tensor): A tensor of tokenized input IDs.

    Returns:
        tuple: A tuple `(logits, hidden_states, attentions)` where:
        logits (torch.Tensor): Output logits from the model.
        hidden_states (tuple of torch.Tensor): Hidden states from each model layer.
        attentions (tuple of torch.Tensor): Attention weights from each model layer.
    """
    # kwargs = {
    #     "input_ids": tok_in,
    #     "use_cache": False,
    #     "past_key_values": None,
    #     "output_attentions": True,
    #     "output_hidden_states": True,
    #     "return_dict": True,
    # }
    kwargs = {
        "input_ids": tok_in,
        "output_attentions": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    
    with torch.no_grad():
        output = model(**kwargs)
    return output.logits, output.hidden_states, output.attentions

#返回logits
def get_logits(model, tok_in):
    """Get only the logits from the model forward pass.

    Args:
        model: A pretrained model compatible with the transformers API.
        tok_in (torch.Tensor): A tensor of tokenized input IDs.

    Returns:
        torch.Tensor: The output logits of the model for the given input.
    """
    kwargs = {
        "input_ids": tok_in,
        "use_cache": False,
        "past_key_values": None,
        "output_attentions": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    with torch.no_grad():
        output = model(**tok_in)
    return output.logits

#返回hidden_states
def get_hidden_acts(model, tok_in):
    """Get hidden states (activations) from the model forward pass.

    Args:
        model: A pretrained model compatible with the transformers API.
        tok_in (torch.Tensor): A tensor of tokenized input IDs.

    Returns:
        tuple of torch.Tensor: The hidden states from each layer of the model.
    """
    kwargs = {
        "input_ids": tok_in,
        "use_cache": False,
        "past_key_values": None,
        "output_attentions": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    with torch.no_grad():
        output = model(**tok_in, output_attentions=True)
    return output.hidden_states

##返回attentions
def get_attentions(model, tok_in):
    """Get attention matrices from the model forward pass.

    Args:
        model: A pretrained model compatible with the transformers API.
        tok_in (torch.Tensor): A tensor of tokenized input IDs.

    Returns:
        tuple of torch.Tensor: The attention matrices from each layer and head.
    """
    kwargs = {
        "input_ids": tok_in,
        # "use_cache": False,
        # "past_key_values": None,
        "output_attentions": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    with torch.no_grad():
        output = model(**tok_in, output_attentions=True)
    return output.attentions
    # outputs_short = model(**tok_in, output_attentions=True)
    # return outputs_short.attentions

#计算中心奇异值
def centered_svd_val(Z, alpha=0.001):
    """Compute the mean log singular value of a centered covariance matrix.

    This function centers the data and computes the singular value decomposition
    (SVD) of the resulting covariance matrix. It then returns the mean of the
    log singular values, regularized by `alpha`.

    Args:
        Z (torch.Tensor): A 2D tensor representing features hidden acts.
        alpha (float, optional): Regularization parameter added to the covariance matrix.
            Defaults to 0.001.

    Returns:
        float: The mean of the log singular values of the centered covariance matrix.
    """
    # assumes Z is in full precision
    J = torch.eye(Z.shape[0]) - (1 / Z.shape[0]) * torch.ones(Z.shape[0], Z.shape[0])
    Sigma = torch.matmul(torch.matmul(Z.t(), J), Z)
    #print(Sigma)
    Sigma = Sigma + alpha * torch.eye(Sigma.shape[0])
    if not torch.isfinite(Sigma).all():
        #print("Input matrix contains non-finite values.")
        #print("Non-finite values:", ~torch.isfinite(Sigma))
        Sigma[~torch.isfinite(Sigma)] = 0  # 将非有限值替换为0
    svdvals = torch.linalg.svdvals(Sigma)
    eigscore = torch.log(svdvals).mean()
    return eigscore

#返回给定layer的奇异值·
def get_svd_eval(hidden_acts, layer_num=15, tok_lens=[], use_toklens=True):
    """Evaluate hidden states at a given layer using SVD-based scoring.

    For each sample, this function extracts the hidden states at a specified layer,
    optionally slices them according to `tok_lens`, and computes the SVD-based score.

    Args:
        hidden_acts (list): A list of tuples, each containing hidden states for all layers
            for a single sample.   #一个列表，列表中的元素是一个元组，元组内是每个样本的所有层的隐层状态
        layer_num (int, optional): The layer index to evaluate. Defaults to 15.
        tok_lens (list, optional): A list of (start, end) indices for each sample to slice
            the hidden states. Defaults to [].
        use_toklens (bool, optional): Whether to slice the hidden states using `tok_lens`.
            Defaults to True.

    Returns:
        np.array: An array of SVD-based scores for each sample.
    """
    svd_scores = []
    for i in range(len(hidden_acts)):
        Z = hidden_acts[i][layer_num]  #选择样本i的第layer_num层
        if torch.isnan(Z).any() or torch.isinf(Z).any():
            print("Warning: Z contains NaN or Inf values. Replacing them with zeros.")
            #Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
            print('第',i,'个样本，','第',layer_num,'层')
            print(Z)

        if use_toklens and tok_lens[i]:
            i1, i2 = tok_lens[i][0], tok_lens[i][1]   #对样本i的token进行切割
            Z = Z[i1:i2, :]  #选择样本i的i1-i2部分的token进行分析

        Z = torch.transpose(Z, 0, 1)
        svd_scores.append(centered_svd_val(Z).item())
    # print("Sigma matrix shape:",Z.shape[1])
    return np.stack(svd_scores)

#返回注意力层的奇异值
def get_attn_eig_prod(attns, layer_num=15, tok_lens=[], use_toklens=True):
    """Compute an eigenvalue-based attention score by analyzing attention matrices.

    This function takes the attention matrices of a given layer and for each sample,
    computes the mean log of the diagonal elements (assumed to be eigenvalues) across
    all attention heads. Slices are applied if `tok_lens` is used.

    Args:
        attns (list): A list of tuples, each containing attention matrices for all layers
            and heads for a single sample.
        layer_num (int, optional): The layer index to evaluate. Defaults to 15.
        tok_lens (list, optional): A list of (start, end) indices for each sample to slice
            the attention matrices. Defaults to [].
        use_toklens (bool, optional): Whether to slice the attention matrices using `tok_lens`.
            Defaults to True.

    Returns:
        np.array: An array of computed attention-based eigenvalue scores for each sample.
    """
    attn_scores = []

    for i in range(len(attns)):  # iterating over number of samples
        eigscore = 0.0
        for attn_head_num in range(len(attns[i][layer_num])):  # iterating over number of attn heads
            # attns[i][layer_num][j] is of size seq_len x seq_len
            #print(attns[i])
            #print(attns[i][layer_num])
            Sigma = attns[i][layer_num][attn_head_num]  #第i个样本的第layer_num层的第attn_head_num个注意力头
            #print(Sigma)

            if use_toklens and tok_lens[i]:
                i1, i2 = tok_lens[i][0], tok_lens[i][1]
                Sigma = Sigma[i1:i2, i1:i2]

            eigscore += torch.log(torch.diagonal(Sigma, 0)).mean()   #.mean()  #对每一层的多个头做平均
            #print(eigscore)
        
        attn_scores.append(eigscore.item())
        print(attn_scores)
    return np.stack(attn_scores)

def get_Rel_Prom_Res(hidden_acts, attns, layer_num,tok_lens, use_toklens):

    """Compute an eigenvalue-based attention score by analyzing attention matrices.

    This function takes the attention matrices of a given layer and for each sample,
    computes the mean log of the diagonal elements (assumed to be eigenvalues) across
    all attention heads. Slices are applied if `tok_lens` is used.

    Args:
        attns (list): A list of tuples, each containing attention matrices for all layers
            and heads for a single sample.
        layer_num (int, optional): The layer index to evaluate. Defaults to 15.
        tok_lens (list, optional): A list of (start, end) indices for each sample to slice
            the attention matrices. Defaults to [].
        use_toklens (bool, optional): Whether to slice the attention matrices using `tok_lens`.
            Defaults to True.

    Returns:
        np.array: An array of computed attention-based eigenvalue scores for each sample.
    """
    Rel_scores = []
    for i in range(len(hidden_acts)):
        Z = hidden_acts[i][layer_num]  #选择样本i的第layer_num层

        i1, i2 = tok_lens[i][0], tok_lens[i][1]   #对样本i的token进行切割
        Z_last = Z[i2-1, :]  #选择最后一个token的hidden states
        
        cos_sim=0
        for attn_head_num in range(len(attns[i][layer_num])):  # iterating over number of attn heads
            # attns[i][layer_num][j] is of size seq_len x seq_len
            Sigma = attns[i][layer_num][attn_head_num]  #第i个样本，layer_num层，第attn_head_num个头

            i1, i2 = tok_lens[i][0], tok_lens[i][1]     
            Sigma = Sigma[0:i1, 0:i1]   #此处只选择prompt中最重要的2/3个token，与回复的token计算cos
            # 提取对角线元素作为特征值
            eigenvalues = torch.diagonal(Sigma, 0)

            # 计算需要选择的特征值数量（前2/3个）
            num_select = int(np.ceil((1/3) * i1))

            # 找到前2/3个最大的特征值的下标
            indices = np.argsort(eigenvalues)[-num_select:]
                #print(indices)
                #print(Z.shape)
                #print(indices)
                # 将特征值下标对应到tok_in的位置
                # cos=0
                # for j in indices:
                #     selected_tok_in = Z[j,:]
                #     cos=cos+util.cos_sim(selected_tok_in,Z_last)
                # cos_sim=cos_sim+cos/len(indices)

            mean_selected = torch.mean(Z[indices,:], dim=0)
            cos_sim=cos_sim+util.cos_sim(mean_selected,Z_last)
                #print('平均相似度为：',util.cos_sim(mean_selected,Z_last))

        cos_sim=cos_sim/len(attns[i][layer_num])
        Rel_scores.append(cos_sim.item())
        #print('注意力头的个数:',len(attns[i][layer_num]))
    return np.stack(Rel_scores)

#返回困惑度
def perplexity(logits, tok_ins, tok_lens, min_k=None):
    """Compute the perplexity of model predictions for given tokenized inputs.

    This function computes the perplexity by taking the negative log probability
    of the correct tokens and exponentiating the mean. If `min_k` is provided,
    it filters the lowest probabilities to compute a restricted perplexity.

    Args:
        logits: A list or array of model logits (samples x seq_len x vocab_size).
        tok_ins: A list of tokenized input IDs for each sample.
        tok_lens (list): A list of (start, end) indices specifying the portion of the
            sequence to evaluate.
        min_k (float, optional): A fraction of tokens to consider from the lowest
            probabilities. If not None, only these tokens are considered.

    Returns:
        np.array: An array of perplexity values for each sample.
    """
    softmax = torch.nn.Softmax(dim=-1)
    ppls = []

    for i in range(len(logits)):
        i1, i2 = tok_lens[i][0], tok_lens[i][1]   #response的起始位置
        pr = torch.log(softmax(logits[i]))[torch.arange(i1, i2) - 1, tok_ins[i][0, i1:i2]]
        if min_k is not None:
            pr = torch.topk(pr, k=int(min_k * len(pr)), largest=False).values
        ppls.append(torch.exp(-pr.mean()).item())

    return np.stack(ppls)

#返回熵
def logit_entropy(logits, tok_lens, top_k=None):
    """Compute the entropy of the model's output distribution over tokens.

    For each sample, this function computes the entropy of the softmax distribution
    over predicted tokens. If `top_k` is provided, only the top K predictions are considered
    when computing entropy.

    Args:
        logits: A list or array of model logits (samples x seq_len x vocab_size).
        tok_lens (list): A list of (start, end) indices specifying the portion of the
            sequence to evaluate.
        top_k (int, optional): Number of top tokens to consider for computing the entropy.
            If None, considers all tokens.

    Returns:
        np.array: An array of entropy values for each sample.
    """
    softmax = torch.nn.Softmax(dim=-1)
    scores = []

    for i in range(len(logits)):
        i1, i2 = tok_lens[i][0], tok_lens[i][1]
        if top_k is None:
            l = softmax(torch.tensor(logits[i]))[i1:i2]
            scores.append((-l * torch.log(l)).mean())
        else:
            l = logits[i][i1:i2]
            l = softmax(torch.topk(l, top_k, 1).values)
            scores.append((-l * torch.log(l)).mean())

    return np.stack(scores)

#返回窗口熵
def window_logit_entropy(logits, tok_lens, top_k=None, w=1):
    """Compute the maximum average entropy in windows of tokens.

    This function computes the entropy as in `logit_entropy`, but applies a sliding window
    of width `w` over the token dimension and returns the maximum mean entropy found.

    Args:
        logits: A list or array of model logits (samples x seq_len x vocab_size).
        tok_lens (list): A list of (start, end) indices specifying the portion of the
            sequence to evaluate.
        top_k (int, optional): Number of top tokens to consider for computing the entropy.
            If None, considers all tokens.
        w (int, optional): Window size to compute local entropy. Defaults to 1.

    Returns:
        np.array: An array of maximum windowed entropy values for each sample.
    """
    softmax = torch.nn.Softmax(dim=-1)
    scores = []

    for i in range(len(logits)):
        i1, i2 = tok_lens[i][0], tok_lens[i][1]
        if top_k is None:
            l = softmax(logits[i])[i1:i2]
        else:
            l = torch.tensor(logits[i])[i1:i2]
            l = softmax(torch.topk(l, top_k, 1).values)
        windows = torch.max((-l * torch.log(l)).mean(1).unfold(0, w, w).mean(1))
        scores.append(windows.item())

    return np.stack(scores)


def def_dict_value():
    return []
