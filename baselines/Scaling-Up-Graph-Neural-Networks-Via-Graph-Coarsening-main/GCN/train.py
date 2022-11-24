import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net, StandardGCN
import numpy as np
from utils import load_data, coarsening
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--coarsening_ratio', type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='variation_neighborhoods')

    parser.add_argument('--num_layers', type=int, default=3, choices=range(2, 6),
                        help='Number of layers of the model.')
    parser.add_argument('--hidden_channels', type=int, default=128, choices=[64, 128, 256],
                        help='Number of hidden channels.')
    parser.add_argument('--batchnorm', type=bool, default=True, choices=[True, False],
                        help='Whether to use batch normalization.')
    parser.add_argument('--dropout', type=float, default=0, choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help='Drop out rate.')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'Sigmoid', 'None'],
                        help='Activation function to use.')
    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Start coarsening...')
    args.num_features, args.num_classes, candidate, C_list, Gc_list = coarsening(args.dataset, 1-args.coarsening_ratio, args.coarsening_method)
    print('Finish coarsening!')
    # model = Net(args).to(device)
    model = StandardGCN(args.num_features, args.hidden_channels, args.num_classes,
                        args.num_layers, args.batchnorm, args.dropout, args.activation).to(device)
    print('Model loaded!')
    all_acc = []

    for i in range(args.runs):
        print('Run_{}'.format(i))
        data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
            args.dataset, candidate, C_list, Gc_list, args.experiment)
        data = data.to(device)
        coarsen_features = coarsen_features.to(device)
        coarsen_train_labels = coarsen_train_labels.to(device)
        coarsen_train_mask = coarsen_train_mask.to(device)
        coarsen_val_labels = coarsen_val_labels.to(device)
        coarsen_val_mask = coarsen_val_mask.to(device)
        coarsen_edge = coarsen_edge.to(device)
        print('Data loaded!')

        if args.normalize_features:
            coarsen_features = F.normalize(coarsen_features, p=1)
            data.x = F.normalize(data.x, p=1)

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_loss = float('inf')
        val_loss_history = []
        for epoch in range(args.epochs):

            model.train()
            optimizer.zero_grad()
            out = model(coarsen_features, coarsen_edge)
            loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
            loss.backward()
            optimizer.step()
            print('train loss: {}'.format(loss))

            model.eval()
            pred = model(coarsen_features, coarsen_edge)
            val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()
            print('val loss: {}'.format(val_loss))

            if val_loss < best_val_loss and epoch > args.epochs // 2:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

            val_loss_history.append(val_loss)
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = tensor(val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

            # eval on true graph
            model.eval()
            pred = model(data.x, data.edge_index).max(1)[1]
            test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
            print(test_acc)

        model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
        model.eval()
        pred = model(data.x, data.edge_index).max(1)[1]
        test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())

        # train_acc = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()) / int(data.train_mask.sum())
        # val_acc = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()) / int(data.val_mask.sum())
        print(test_acc)
        all_acc.append(test_acc)
        # print(f'Epoch: {epoch:04d}, '
        #       f'Loss: {loss:012.3f}, '
        #       f'Train: {100 * train_acc:05.2f}%, '
        #       f'Valid: {100 * valid_acc:05.2f}%, '
        #       f'Test: {100 * test_acc:05.2f}%, ')

    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

