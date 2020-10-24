# Correct and Smooth OGB submission

This directory contains OGB submissions. All hyperparameters were tuned on the validation set with optuna, except for products, which was hand tuned.


## Arxiv

### Label Propagation (0 params):
```
python run_experiments.py --dataset arxiv --method lp

Valid acc: 0.7013658176448874
Test acc: 0.6832294302820814
```

### Plain Linear C&S (5160 params)
```
python gen_models.py --model plain --epochs 1000    
python run_experiments.py --dataset arxiv --method plain

Valid acc -> Test acc
Args []: 73.00 ± 0.01 -> 71.23 ± 0.01
```

### Linear C&S (15400 params)
```
python gen_models.py --model linear --epochs 1000
python run_experiments.py --dataset arxiv --method linear

Valid acc -> Test acc
Args []: 73.68 ± 0.04 -> 72.21 ± 0.02
```

### MLP C&S (175656 params)
```
python gen_models.py --model mlp --epochs 500
python run_experiments.py --dataset arxiv --method mlp

Valid acc -> Test acc
Args []: 73.91 ± 0.15 -> 73.12 ± 0.12
```


## Products

### Label Propagation (0 params):
```
python run_experiments.py --dataset products --method lp

Valid acc:  0.9090608549703736
Test acc: 0.7434145274640762
```