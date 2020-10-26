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
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    args = parser.parse_args()
    
    dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}')
    data = dataset[0]
    
    adj, D_isqrt = process_adj(data)
    normalized_adjs = gen_normalized_adjs(adj, D_isqrt)
    DAD, DA, AD = normalized_adjs
    evaluator = Evaluator(name=f'ogbn-{args.dataset}')
    
    split_idx = dataset.get_idx_split()
  
    def eval_test(result, idx=split_idx['test']):
        return evaluator.eval({'y_true': data.y[idx],'y_pred': result[idx].argmax(dim=-1, keepdim=True),})['acc']
    
    if args.dataset == 'arxiv':
        lp_dict = {
            'idxs': ['train'],
            'alpha': 0.9,
            'num_propagations': 50,
            'A': AD,
        }
        plain_dict = {
            'train_only': True,
            'alpha1': 0.87,
            'A1': AD,
            'num_propagations1': 50,
            'alpha2': 0.81,
            'A2': DAD,
            'num_propagations2': 50,
            'display': False,
        }
        plain_fn = double_correlation_autoscale
        
        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9988673963255859, 'alpha2': 0.7942279952481052, 'A1': 'DA', 'A2': 'AD'} 
        gets you to 72.64
        """
        linear_dict = {
            'train_only': True,
            'alpha1': 0.98, 
            'alpha2': 0.65, 
            'A1': AD, 
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        linear_fn = double_correlation_autoscale
        
        """
        If you tune hyperparameters on test set
        {'alpha1': 0.9956668128133523, 'alpha2': 0.8542393515434346, 'A1': 'DA', 'A2': 'AD'}
        gets you to 73.35
        """
        mlp_dict = {
            'train_only': True,
            'alpha1': 0.9791632871592579, 
            'alpha2': 0.7564990804200602, 
            'A1': DA, 
            'A2': AD,
            'num_propagations1': 50,
            'num_propagations2': 50,
            'display': False,
        }
        mlp_fn = double_correlation_autoscale  
        
        gat_dict = {
            'labels': ['train'],
            'alpha': 0.8, 
            'A': DAD,
            'num_propagations': 50,
            'display': False,
        }
        gat_fn = only_outcome_correlation

        
    elif args.dataset == 'products':
        lp_dict = {
            'idxs': ['train'],
            'alpha': 0.5,
            'num_propagations': 50,
            'A': DAD,
        }
        
        plain_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.9, 
            'scale': 20.0, 
            'A1': DAD, 
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        plain_fn = double_correlation_fixed
        
        linear_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.9, 
            'scale': 20.0, 
            'A1': DAD, 
            'A2': DAD,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        linear_fn = double_correlation_fixed
        
        mlp_dict = {
            'train_only': True,
            'alpha1': 1.0,
            'alpha2': 0.8, 
            'scale': 10.0, 
            'A1': DAD, 
            'A2': DA,
            'num_propagations1': 50,
            'num_propagations2': 50,
        }
        mlp_fn = double_correlation_fixed




    model_outs = glob.glob(f'models/{args.dataset}_{args.method}/*.pt')
    
    if args.method == 'lp':
        out = label_propagation(data, split_idx, **lp_dict)
        print('Valid acc: ', eval_test(out, split_idx['valid']))
        print('Test acc:', eval_test(out, split_idx['test']))
        return
    
    get_orig_acc(data, eval_test, model_outs, split_idx)
    while True:
        if args.method == 'plain':
            evaluate_params(data, eval_test, model_outs, split_idx, plain_dict, fn = plain_fn)
        elif args.method == 'linear':
            evaluate_params(data, eval_test, model_outs, split_idx, linear_dict, fn = linear_fn)
        elif args.method == 'mlp':
            evaluate_params(data, eval_test, model_outs, split_idx, mlp_dict, fn = mlp_fn)
        elif args.method == 'gat':
            evaluate_params(data, eval_test, model_outs, split_idx, gat_dict, fn = gat_fn) 
#         import pdb; pdb.set_trace()
        break
        
#     name = f'{args.experiment}_{args.search_type}_{args.model_dir}'
#     setup_experiments(data, eval_test, model_outs, split_idx, normalized_adjs, args.experiment, args.search_type, name, num_iters=300)
    
#     return

    
if __name__ == "__main__":
    main()
