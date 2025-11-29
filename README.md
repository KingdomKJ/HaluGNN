
# HaluGNN: Hallucination Detection in Large Language Models Using Graph Neural Network

## Project Overview

HaluGNN is a Graph Neural Network (GNN) based framework for detecting hallucinations in Large Language Models (LLMs). The project constructs semantic graphs from LLM internal states (hidden representations and attention mechanisms) to identify unreliable model outputs.

## Installation

```bash
# Install dependencies
pip install torch torch-geometric transformers sentence-transformers scikit-learn networkx matplotlib pandas numpy tqdm
```

## Project Structure

```
HaluGNN/
├── model/                          # model_path
├── data/                           # Dataset directory
│   ├── TruthfulQA/                 # TruthfulQA dataset
│   ├── TriviaQA/                   # TriviaQA dataset  
│   ├── sciQ/                       # SciQ dataset
│   └── ragtruth/                   # RAGTruth dataset
├── training_scripts/               # Training scripts
│   ├── truthfulQA_llama2.py       # TruthfulQA + LLaMA2
│   ├── TriviaQA_llama2.py         # TriviaQA + LLaMA2
│   ├── sciQ_llama2.py             # SciQ + LLaMA2
│   ├── truthfulQA_Qwen.py         # TruthfulQA + Qwen
│   ├── TriviaQA_Qwen.py           # TriviaQA + Qwen
│   ├── sciQ_Qwen.py               # SciQ + Qwen
│   └── ragtruth_llama3.py         # RAGTruth + LLaMA3
├── utils/                          # Utilities
│   ├── common_utils.py            # Common functions
│   ├── utils_TriviaQA.py          # TriviaQA processing
│   ├── utils_SciQ.py              # SciQ processing
│   └── run_detection_combined.py  # Main detection script
|   └── ..........
└── README.md
```

## Quick Start

### Data Preprocessing

```bash
# Extract features from LLM internal states
python run_detection_combined.py --dataset TruthfulQA --model llama --n_samples 100
```

### Model Training

```bash
# Train on different datasets and models
python truthfulQA_llama2.py --type raw
python TriviaQA_Qwen.py --type screen
...
```

### Parameters

- `--type`: Data type (`raw` or `screen`)
- `--gnn`: GNN variant (`_cut` or others)
- `--dataset`: Dataset name
- `--model`: LLM name
- `--n_samples`: Number of samples

## Model Architecture

### Key Components

- **Graph Construction**: Hidden states as nodes, attention weights as edges
- **Graph Convolution**: Two-layer GraphConv with ReLU activation
- **Pooling**: Global mean pooling for graph-level classification
- **Regularization**: Dropout for preventing overfitting

## Supported Models and Datasets

### Large Language Models
- LLaMA-2-7B-chat
- LLaMA-3-8B-Instruct
- Qwen2-7B-Instruct

### Evaluation Datasets
- TruthfulQA
- TriviaQA
- SciQ
- RAGTruth

## Evaluation Metrics

The framework supports comprehensive evaluation:

```python
from common_utils import get_roc_scores

# Calculate evaluation metrics
```

**Metrics Included:**
- Accuracy
- F1-Score
- AUC-ROC
- TPR@5%FPR (True Positive Rate at 5% False Positive Rate)
- AUROC

## Interpretability Features

### GNNExplainer Integration

```python
from torch_geometric.explain import Explainer, GNNExplainer


### Visualization Tools

- **t-SNE Visualization**: Graph embedding analysis
- **Node Importance**: Token-level contribution analysis
- **Edge Importance**: Semantic relationship analysis
- **Attention Visualization**: LLM attention pattern analysis

## Efficiency Analysis


### Computational Complexity

- **Time Complexity**: O(N + E) for N nodes and E edges
- **Space Complexity**: O(Nd + N²) for node features and adjacency matrix

## Advanced Usage

### Custom Dataset Integration

1. Create data processor in `utils/` directory
2. Implement required functions:
```python
def get_custom_data(n_samples=200):
    # Your data loading logic
    return train_data, test_data

def get_scores_dict(model_name_or_path, data, mt_list, args):
    # Your scoring logic
    return scores, indiv_scores, labels, hidden_acts, att_acts
```

3. Register in `common_utils.py`:
```python
elif args.dataset == "your_dataset":
    import utils_your_dataset
    get_data = utils_your_dataset.get_custom_data
    get_scores_dict = utils_your_dataset.get_scores_dict
```

### Model Configuration

Adjust model parameters based on your needs:

```python
model = GNNModel(
    input_dim=4096,      # LLaMA-2 hidden dimension
    # input_dim=3584,    # Qwen hidden dimension
    hidden_dim=512,      # Hidden layer size
    output_dim=2         # Binary classification
)
```

## Citation

If you use HaluGNN in your research, please cite:

```bibtex
@article{halugnn2025,
  title={HaluGNN: Hallucination Detection in Large Language Models Using Graph Neural Network},
  author={Linggang Kong, Yunlong Zhang, Xiaofeng Zhong, Haoran Fu, Yongjie Wang, Huijun Liu},
  journal={Expert Systems with Applications},
  year={2025}
}
```

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Email: konglinggang@nudt.edu.cn

## Acknowledgments

- Thanks to the open-source community for the foundational models and datasets
- Inspired by recent advances in graph neural networks and LLM interpretability
```
