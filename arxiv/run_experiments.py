import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
import glob
from copy import deepcopy
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
import random
from outcome_correlation import *

def main():
    parser = argparse.ArgumentParser(description='Outcome Correlations)')
    parser.add_argument('--model_dir', type=str, default='default')
    parser.add_argument('--experiment', type=str, default='origacc')
    parser.add_argument('--search_type', type=str, default='none')
    args = parser.parse_args()
    model_outs = glob.glob(f'models/{args.model_dir}/*.pt')
    
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    
    adj, D_isqrt = process_adj(data)
    normalized_adjs = gen_normalized_adjs(adj, D_isqrt)
    DAD, DA, AD = normalized_adjs
    evaluator = Evaluator(name='ogbn-arxiv')
    
    split_idx = dataset.get_idx_split()
    
    def eval_test(result, idx=split_idx['test']):
        return evaluator.eval({'y_true': data.y[idx],'y_pred': result[idx].argmax(dim=-1, keepdim=True),})['acc']
    
    param_dict = {
        'idxs': ['train'],
        'alpha': 0.9,
        'num_propagations': 50,
        'A': AD,
    }
    print('Valid acc: ', eval_test(label_propagation(data, split_idx, **param_dict), split_idx['valid']))
    print('Valid acc: ', eval_test(label_propagation(data, split_idx, **param_dict), split_idx['test']))
    return
    name = f'{args.experiment}_{args.search_type}_{args.model_dir}'
    setup_experiments(data, eval_test, model_outs, split_idx, normalized_adjs, args.experiment, args.search_type, name, num_iters=300)
    
    return

    
if __name__ == "__main__":
    main()