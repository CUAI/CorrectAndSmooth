# Correct and Smooth OGB submission

Currently, this directory only contains label propagation baselines for OGB.


## Arxiv

### Label Propagation (0 params):
```
python run_experiments.py --dataset arxiv --method lp

Valid acc: 0.7013658176448874
Test acc: 0.6832294302820814
```
### Plain C&S (5160 params)
```
python gen_models.py --model plain --epochs 1000    
python run_experiments.py --dataset arxiv --method plain

Valid acc -> Test acc
Args []: 73.00 ± 0.01 -> 71.23 ± 0.01
```

## Products

### Label Propagation (0 params):
```
python run_experiments.py --dataset products --method lp

Valid acc:  0.9090608549703736
Test acc: 0.7434145274640762
```