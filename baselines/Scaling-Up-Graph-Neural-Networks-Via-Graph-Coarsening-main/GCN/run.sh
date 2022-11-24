#To overcome over-fitting, we fine-tuned the hyperparameter settings in the few-shot regime.
#To overcome over-smoothing, we fine-tuned the hyperparameter settings when the coarsening ratio is 0.1.
#example, the coarsening ratio is 0.5
python train.py --dataset ogbn-arxiv --experiment fixed --coarsening_ratio 0.5 --runs 1 --epochs 200 --lr 1e-3 --hidden_channels 128
#python train.py --dataset cora --experiment fixed --coarsening_ratio 0.5
#python train.py --dataset cora --experiment few --epoch 100 --coarsening_ratio 0.5
#python train.py --dataset citeseer --experiment fixed --epoch 200 --coarsening_ratio 0.5
#python train.py --dataset pubmed --experiment fixed --epoch 200 --coarsening_ratio 0.5
#python train.py --dataset pubmed --experiment few --epoch 60 --coarsening_ratio 0.5
#python train.py --dataset dblp --experiment random --epoch 50 --coarsening_ratio 0.5
#python train.py --dataset Physics --experiment random --epoch 200 --lr 0.001 --weight_decay 0 --coarsening_ratio 0.5
