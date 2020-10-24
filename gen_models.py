import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import numpy as np


from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from outcome_correlation import prepare_folder
import glob
import os
import shutil

from logger import Logger

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):    
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)

class MLPLinear(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLPLinear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, adj_t):
        return F.log_softmax(self.lin(x), dim=-1)

    
def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y, adj, split_idx, evaluator):
    model.eval()

    out = model(x, adj)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

    
        
            
def main():
    parser = argparse.ArgumentParser(description='gen_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--model_dir', type=str)

    args = parser.parse_args()
    print(args)

    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}',transform=T.ToSparseTensor())
    
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    
    x = data.x
    
    if args.use_node_embedding:
        embedding = preprocess(Planetoid(root='.', name=dataset_name)[0], 'sgc', post_fix=dataset_name)

        embedding = torch.load('embedding.pt', map_location=device)
        diffusion = torch.load('diffusion.pt', map_location=device)
        x = torch.cat([x, embedding, diffusion], dim=-1)


    x = x.to(device)
    adj_t = data.adj_t.to(device)
    y_true = data.y.to(device)
    
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    if args.model == 'mlp':
        model = MLP(x.size(-1),256, dataset.num_classes, 3, 0.5).cuda()
    elif args.model=='linear':
        if dataset == 'arxiv':
            x = torch.cat([x, preprocess(data, 'diffusion')])
        x = torch.cat([x, preprocess(data, 'spectral', post_fix=dataset_name)])
        model = MLPLinear(x.size(-1), 256, dataset.num_classes).cuda()
        
    elif args.model=='plain':
        model = MLPLinear(x.size(-1), 256, dataset.num_classes).cuda()

    
    model_dir = prepare_folder(f'{args.dataset}_{args.model}', model)

    
    evaluator = Evaluator(name=f'ogbn-{args.dataset}')
    logger = Logger(args.runs, args)
    
    for run in range(args.runs):
        print(sum(p.numel() for p in model.parameters()))
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_valid = 0
        best_out = None
        
        for epoch in range(1, args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x, adj_t)[train_idx]
            loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
            result = test(model, x, y_true, adj_t, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_out = model(x, adj_t).detach().exp()
        
            print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            logger.add_result(run, result)
            loss.backward()
            optimizer.step()
        logger.print_statistics(run)
        torch.save(best_out, f'{model_dir}/{run}.pt')

    logger.print_statistics()




if __name__ == "__main__":
    main()
