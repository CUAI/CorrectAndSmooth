import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
import glob



from copy import deepcopy
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import optuna

from logger import Logger
import random
import shutil
import glob
from collections.abc import Iterable
import joblib



class SimpleLogger(object):
    def __init__(self, desc, param_names, num_values=2):
        self.results = defaultdict(dict)
        self.param_names = tuple(param_names)
        self.used_args = list()
        self.desc = desc
        self.num_values = num_values
    
    def add_result(self, run, args, values): 
        """Takes run=int, args=tuple, value=tuple(float)"""
        assert(len(args) == len(self.param_names))
        assert(len(values) == self.num_values)
        self.results[run][args] = values
        if args not in self.used_args:
            self.used_args.append(args)
    
    def get_best(self, top_k=1):
        all_results = []
        for args in self.used_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)[-1]
            results_std = results.std(dim=0)

            all_results.append((args, results_mean))
        results = sorted(all_results, key=lambda x: x[1], reverse=True)[:top_k]
        return [i[0] for i in results]
            
    def prettyprint(self, x):
        if isinstance(x, float):
            return '%.2f' % x
        return str(x)
        
    def display(self, args = None):
        
        disp_args = self.used_args if args is None else args
        if len(disp_args) > 1:
            print(f'{self.desc} {self.param_names}, {len(self.results.keys())} runs')
        for args in disp_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)
            results_std = results.std(dim=0)
            res_str = f'{results_mean[0]:.2f} ± {results_std[0]:.2f}'
            for i in range(1, self.num_values):
                res_str += f' -> {results_mean[i]:.2f} ± {results_std[1]:.2f}'
            print(f'Args {[self.prettyprint(x) for x in args]}: {res_str}')
        if len(disp_args) > 1:
            print()

def process_adj(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    return adj, deg_inv_sqrt

def gen_normalized_adjs(adj, D_isqrt):
    DAD = D_isqrt.view(-1,1)*adj*D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1)*adj
    AD = adj*D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD

def gen_normalized_adj(adj, pw): # pw = 0 is D^-1A, pw=1 is AD^-1
    deg = adj.sum(dim=1).to(torch.float)
    front  = deg.pow(-(1-pw))
    front[front == float('inf')] = 0
    back  = deg.pow(-(pw))
    back[back == float('inf')] = 0
    return (front.view(-1,1)*adj*back.view(1,-1))

def model_load(file, device='cpu'):
    result = torch.load(file, map_location='cpu')
    run = get_run_from_file(file)
    try:
        split = torch.load(f'{file}.split', map_location='cpu')
    except:
        split = None
    
    mx_diff = (result.sum(dim=-1) - 1).abs().max()
    if mx_diff > 1e-1:
        print(f'Max difference: {mx_diff}')
        print("model output doesn't seem to sum to 1. Did you remember to exp() if your model outputs log_softmax()?")
        raise Exception
    if split is not None:
        return (result, split), run
    else:
        return result, run

def get_labels_from_name(labels, split_idx):
    if isinstance(labels, list):
        labels = list(labels)
        if len(labels) == 0:
            return torch.tensor([])
        for idx, i in enumerate(list(labels)):
            labels[idx] = split_idx[i]
        residual_idx = torch.cat(labels)
    else:
        residual_idx = split_idx[labels]
    return residual_idx
        
def pre_residual_correlation(labels, model_out, label_idx):
    """Generates the initial labels used for residual correlation"""
    labels = labels.cpu()
    labels[labels.isnan()] = 0
    labels = labels.long()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    y = torch.zeros((n, c))
    y[label_idx] = F.one_hot(labels[label_idx],c).float().squeeze(1) - model_out[label_idx]
    return y

def pre_outcome_correlation(labels, model_out, label_idx, thresh):
    """Generates the initial labels used for outcome correlation"""

    labels = labels.cpu()
    model_out = model_out.cpu()
    label_idx = label_idx.cpu()
    c = labels.max() + 1
    n = labels.shape[0]
    y = torch.zeros((n, c))
    model_max = model_out.max(dim=-1)
    high_conf_test_idxs = (model_max[0]> thresh).nonzero(as_tuple=False)
    y[high_conf_test_idxs] = model_out[high_conf_test_idxs]
    
    if len(label_idx) > 0:
        y[label_idx] = F.one_hot(labels[label_idx],c).float().squeeze(1) 
    
    return y

def general_outcome_correlation(adj, y, alpha, num_propagations, post_step, alpha_term, power = 1.0, device='cuda', display=True):
    """general outcome correlation. alpha_term = True for outcome correlation, alpha_term = False for residual correlation"""
    adj = adj.to(device)
    orig_device = y.device
    y = y.to(device)
    result = y.clone()
    for _ in tqdm(range(num_propagations), disable = not display):
        result = alpha * (adj @ result)
        if alpha_term:
            result += (1-alpha)*y
        else:
            result += y
        result = post_step(result)
    return result.to(orig_device)

def label_propagation(data, split_idx, A, alpha, num_propagations, idxs):
    labels = data.y.data
    c = labels.max() + 1
    n = labels.shape[0]
    y = torch.zeros((n, c))
    label_idx = get_labels_from_name(idxs, split_idx)
    y[label_idx] = F.one_hot(labels[label_idx],c).float().squeeze(1) 


    return general_outcome_correlation(A, y, alpha, num_propagations, post_step=lambda x:torch.clamp(x,0,1), alpha_term=True)

def get_autoscale_search_dict(normalized_adjs, idx, train_only):
    DAD, DA, AD = normalized_adjs
    search_dict = {
        'residual_idx': idx,
        'num_propagations1': 50,
        'num_propagations2': 50,
        'alpha1': ('uniform', 0.0, 1.0),
        'alpha2': ('uniform', 0.0, 1.0),
        'train_only': train_only,
        'thresh': 0.0,
        'A1': {'DAD': DAD, 'DA': DA, 'AD': AD},   
        'A2': {'DAD': DAD, 'DA': DA, 'AD': AD},      
    }
    return search_dict

def double_correlation_autoscale(data, model_out, split_idx, A1, alpha1, num_propagations1, A2, alpha2, num_propagations2, scale=1.0, train_only=False, device='cuda', display=True):
    train_idx, valid_idx, test_idx = split_idx
    if train_only:
        label_idx = torch.cat([split_idx['train']])
        residual_idx = split_idx['train']
    else:
        label_idx = torch.cat([split_idx['train'], split_idx['valid']])
        residual_idx = label_idx

        
    y = pre_residual_correlation(labels=data.y.data, model_out=model_out, label_idx=residual_idx)
    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1, post_step=lambda x: torch.clamp(x, -1.0, 1.0), alpha_term=True, display=display, device=device)

    orig_diff = y[residual_idx].abs().sum()/residual_idx.shape[0]
    resid_scale = (orig_diff/resid.abs().sum(dim=1, keepdim=True))
    resid_scale[resid_scale.isinf()] = 1.0
    cur_idxs = (resid_scale > 1000)
    resid_scale[cur_idxs] = 1.0
    res_result = model_out + resid_scale*resid
    res_result[res_result.isnan()] = model_out[res_result.isnan()]
    y = pre_outcome_correlation(labels=data.y.data, model_out=res_result, label_idx = label_idx, thresh=0)
    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2, post_step=lambda x: torch.clamp(x, 0,1), alpha_term=True, display=display, device=device)
    
    return res_result, result

def get_fixed_search_dict(normalized_adjs, idx, train_only):
    DAD, DA, AD = normalized_adjs
    search_dict = {
        'residual_idx': idx,
        'num_propagations1': 50,
        'num_propagations2': 50,
        'alpha1': 1.0,
        'alpha2': ('uniform', 0.0, 1.0),
        'scale': ('uniform', 0.0, 20.0),
        'train_only': train_only,
        'thresh': 0.0,
        'A1': {'DAD': DAD, 'DA': DA, 'AD': AD},   
        'A2': {'DAD': DAD, 'DA': DA, 'AD': AD},      
    }
    return search_dict

def double_correlation_fixed(data, model_out, split_idx, residual_idx, A1, alpha1, num_propagations1, A2, alpha2, num_propagations2, thresh, scale=1.0, train_only=False, device='cuda', display=True):
    train_idx, valid_idx, test_idx = split_idx
    if train_only:
        label_idx = torch.cat([split_idx['train']])
    else:
        label_idx = torch.cat([split_idx['train'], split_idx['valid']])
    residual_idx = get_labels_from_name(residual_idx, split_idx)

    y = pre_residual_correlation(labels=data.y.data, model_out=model_out, label_idx=residual_idx)
    
    fix_y = y[residual_idx].to(device)
    def fix_inputs(x):
        x[residual_idx] = fix_y
        return x
    
    resid = general_outcome_correlation(adj=A1, y=y, alpha=alpha1, num_propagations=num_propagations1, post_step=lambda x: fix_inputs(x), alpha_term=True, display=display, device=device)
    res_result = model_out + scale*resid
    
    y = pre_outcome_correlation(labels=data.y.data, model_out=res_result, label_idx = label_idx, thresh=thresh)

    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2, post_step=lambda x: x.clamp(0, 1), alpha_term=True, display=display, device=device)
    
    return res_result, result


def get_only_outcome_search_dict(normalized_adjs, idx, train_only):
    DAD, DA, AD = normalized_adjs
    search_dict = {
        'num_propagations2': 50,
        'alpha2': ('uniform', 0.0, 1.0),
        'label_idxs': [],
        'thresh': 0.0,
        'A2': {'DAD': DAD, 'DA': DA, 'AD': AD},      
    }
    return search_dict

def only_outcome_correlation(data, model_out, split_idx, A2, alpha2, num_propagations2, thresh, label_idxs, device='cuda', display=True):
    res_result = model_out.clone()
    label_idxs = get_labels_from_name(label_idxs, split_idx)
    y = pre_outcome_correlation(labels=data.y.data, model_out=model_out, label_idx=label_idxs, thresh=thresh)
    result = general_outcome_correlation(adj=A2, y=y, alpha=alpha2, num_propagations=num_propagations2, post_step=lambda x: torch.clamp(x, 0, 1), alpha_term=True, display=display, device=device)
    return res_result, result
    
    
def evaluate_params(data, eval_test, model_outs, split_idx, params, fn=double_correlation_autoscale):
    logger = SimpleLogger('evaluate params', [], 2)

    for out in model_outs:
        model_out, run = model_load(out)
        if isinstance(model_out, tuple):
            model_out, t = model_out
            split_idx = t
        res_result, result = fn(data, model_out, split_idx, **params)
        valid_acc, test_acc = eval_test(result, split_idx['valid']), eval_test(result, split_idx['test'])
        logger.add_result(run, (), (valid_acc, test_acc))
    print('Valid acc -> Test acc')
    logger.display()
    return logger
        


def get_run_from_file(out):
    return int(os.path.splitext(os.path.basename(out))[0])



def get_default_optuna_search_space(train_idx, valid_idx, normalized_adjs):
    DAD, DA, AD = normalized_adjs
    search_dict = {
        'residual_idx': {'valid_idx': valid_idx, 'train_idx': train_idx, 'both': torch.cat([train_idx, valid_idx])},
        'num_propagations1': ('intlog', 1, 50),
        'num_propagations2': ('intlog', 1, 50),
        'alpha1': ('uniform', 0.0, 1.0),
        'alpha2': ('uniform', 0.0, 1.0),
        'thresh': ('uniform', 0.0, 1.0),
        'A1': {'DAD': DAD, 'DA': DA, 'AD': AD},
        'A2': {'DAD': DAD, 'DA': DA, 'AD': AD},   
    }
    
def optuna_search(data, eval_test, model_outs, split_idx, search_space, n_trials=50, fn=double_correlation_autoscale, name='test', obj_idx='test'):
    print('Starting optuna search')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    train_idx, valid_idx, test_idx = split_idx
    fixed_params = {
        'display': False
    }
    search_params = {}
    mapped_params = {}
    for k, v in search_space.items():
        if isinstance(v, tuple):
            search_params[k] = v
        elif isinstance(v, dict):
            search_params[k] = ('dict', list(v.keys()))
            mapped_params[k] = {a: b for a,b in v.items()}
        else:
            fixed_params[k] = v
    
    def objective(trial):
        cur_params = {}
        for k, v in search_params.items():
            if v[0] == 'categorical':
                cur_params[k] = trial.suggest_categorical(k, v[1])
            elif v[0] == 'dict':
                cur_params[k] = trial.suggest_categorical(k, v[1])
                cur_params[k] = mapped_params[k][cur_params[k]]
            elif v[0] == 'uniform':
                cur_params[k] = trial.suggest_uniform(k, v[1], v[2])
            elif v[0] == 'logint':
                cur_params[k] = trial.suggest_int(k, v[1], v[2], log=True)
            elif v[0] == 'discreteuniform':
                cur_params[k] = trial.suggest_discrete_uniform(k, v[1], v[2], v[3])
            elif v[0] == 'loguniform':
                cur_params[k] = trial.suggest_loguniform(k, v[1], v[2])
        
        total_results = []
        res_results = []
        test_results = []
        nonlocal split_idx
        for run, out in enumerate(model_outs):
            model_out, _ = model_load(out)
            if isinstance(model_out, tuple):
                model_out, t = model_out
                split_idx = t
            res_result, result = fn(data, model_out, split_idx, **fixed_params, **cur_params)
            
            total_results.append(eval_test(result, split_idx[obj_idx]))
            trial.report(np.mean(total_results), step=run)
            if trial.should_prune():
                raise optuna.TrialPruned()
            res_results.append(eval_test(res_result, split_idx[obj_idx]))
            if obj_idx != 'test':
                test_results.append((eval_test(res_result, split_idx['test']), eval_test(result, split_idx['test'])))
        trial.set_user_attr('residual accuracy', np.mean(res_results))
        if obj_idx != 'test':
            trial.set_user_attr('test_acc', np.mean(test_results, axis=0))
        print(f'Residual: {np.mean(res_results)}')
        if obj_idx != 'test':
            print(f'Test: {np.mean(test_results, axis=0)}')
        return np.mean(total_results)
    
    def save_best_trial(study, trial):
        pass
#         joblib.dump(study.best_trial, f'logs/{name}.pkl')

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner()) 
    study.optimize(objective, n_trials=n_trials, callbacks=[save_best_trial])
    best_study = study

    print(best_study.best_trial)
    print(best_study.best_trial.user_attrs)
    print(best_study.best_value)
    print(best_study.best_params)
    return best_study

def get_orig_acc(data, eval_test, model_outs, split_idx):
    logger_orig = Logger(len(model_outs))
    for out in model_outs:
        model_out, run = model_load(out)
        if isinstance(model_out, tuple):
            model_out, split_idx = model_out
        test_acc = eval_test(model_out, split_idx['test'])
        print(run, test_acc)
        logger_orig.add_result(run, (eval_test(model_out, split_idx['train']), eval_test(model_out, split_idx['valid']), test_acc))
    print('Original accuracy')
    logger_orig.print_statistics()
    
def setup_experiments(data, eval_test, model_outs, split_idx, normalized_adjs, experiment, search_type, name, num_iters=1000):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    assert len(model_outs) > 0, "No model outs found"
    DAD, DA, AD = normalized_adjs

    get_orig_acc(data, eval_test, model_outs, split_idx)
    if experiment == 'origacc':
        return
    if experiment == 'fixed':
        adjust_params = {
            'scale': 1.0,
            'A1': DA,
        }
        default_params = {
            'alpha2': 0.8,
            'A2': DAD
        }
        search_dict_fn = get_fixed_search_dict
        fn = double_correlation_fixed
    elif experiment == 'fixedscale':
        adjust_params = {
        }
        default_params = {
            'alpha2': 0.9,
            'A1': DA,
            'A2': DAD,
            'scale': 5
        }
        search_dict_fn = get_fixed_search_dict
        fn = double_correlation_fixed
    elif experiment == 'fixedautoscale':
        adjust_params = {
            'scale': 1.0,
            'A1': DA
        }
        default_params = {
            'alpha2': 0.8,
            'A1': DA,
            'A2': DAD,
        }
        search_dict_fn = get_fixed_autoscale_search_dict
        fn = double_correlation_fixed_autoscale
    elif experiment == 'autoscale':
        adjust_params = {
            'scale': 1.0
        }
        default_params = {
            'alpha1': 0.95,
            'alpha2': 0.8,
            'A1': DAD,
            'A2': DAD,
        }
        search_dict_fn = get_autoscale_search_dict
        fn = double_correlation_autoscale
    elif experiment == 'nolabels':
        adjust_params = {
        }
        default_params = {
            'alpha2': 0.8,
            'A2': DAD,
        }
        search_dict_fn = get_only_outcome_search_dict
        fn = only_outcome_correlation

    
    if search_type == 'train':
        # tune on test set; propagate only train
        search_dict = search_dict_fn(normalized_adjs, idx=['train'], train_only=True)
        search_dict = {**search_dict, **adjust_params}

        study = optuna_search(data, eval_test, model_outs, split_idx, search_dict, n_trials=num_iters, fn=fn, name=name)
        
        test_dict = search_dict
        for k, v in test_dict.items():
            if isinstance(v, tuple):
                test_dict[k] = study.best_params[k]
            elif isinstance(v, dict):
                test_dict[k] = search_dict[k][study.best_params[k]]
        logger = evaluate_params(data, eval_test, model_outs, split_idx, test_dict, fn=fn)
    elif search_type == 'validtune':
        # tune on valid set; propagate only train
        search_dict = search_dict_fn(normalized_adjs, idx=['train'], train_only=True)
        search_dict = {**search_dict, **adjust_params}
        study = optuna_search(data, eval_test, model_outs, split_idx, search_dict, n_trials=num_iters, fn=fn, name=name, obj_idx='valid')
        
        test_dict = search_dict
        test_dict["residual_idx"] =  ['train', 'valid']
        test_dict['train_only'] = False

        for k, v in test_dict.items():
            if isinstance(v, tuple):
                test_dict[k] = study.best_params[k]
            elif isinstance(v, dict):
                test_dict[k] = search_dict[k][study.best_params[k]]
    
        logger = evaluate_params(data, eval_test, model_outs, split_idx, test_dict, fn=fn)

    elif search_type == 'all':
        # tune on test set; propagate both
        search_dict = search_dict_fn(normalized_adjs, idx=('categorical', ['train', 'valid', ['train', 'valid']]), train_only=False)
        search_dict = {**search_dict, **adjust_params}
        study = optuna_search(data, eval_test, model_outs, split_idx, search_dict, n_trials=num_iters, fn=fn, name=name)
        test_dict = search_dict
        
        for k, v in test_dict.items():
            if isinstance(v, tuple):
                test_dict[k] = study.best_params[k]
            elif isinstance(v, dict):
                test_dict[k] = search_dict[k][study.best_params[k]]
        logger = evaluate_params(data, eval_test, model_outs, split_idx, test_dict, fn=fn)
    elif search_type in ['none', 'nonetrain']:
        # no tuning
        if search_type == 'none':
            idxs = ['train', 'valid']
            train_only = False
        elif search_type == 'nonetrain':
            idxs = ['train']
            train_only = True
        if experiment != 'nolabels':
            search_dict = search_dict_fn(normalized_adjs, idx=idxs, train_only=train_only)
        
        logger = evaluate_params(data, eval_test, model_outs, split_idx, {**search_dict, **adjust_params, **default_params}, fn =fn)  
#         joblib.dump(logger, f'logs/{name}.pkl')
    print('successfully finished job')

def prepare_folder(name, model):
    model_dir = f'models/{name}'
   
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in model.parameters())}\n')
    return model_dir

