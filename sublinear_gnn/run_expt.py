import argparse
import time
import torch_geometric.transforms as T
import gc
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from model_baselines import StandardGCN, PolyActGCN
from model import SketchGCN
from dataset_baselines import get_baseline_dataloader, get_baseline_test_dataloader
from dataset import get_dataloader, get_test_dataloader
from train_test_baselines import train_baselines, test_baselines
from train_test import train, test
from logger import Logger
from train_utils import *


def main():
    # parse the args
    parser = argparse.ArgumentParser(description='Sketch-GCN Experiments on OGB node classification benchmarks')
    # required args
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['arxiv', 'products'],
                        help='Dataset name.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['Sketch-GCN', 'Poly-Act-GCN', 'Standard-GCN'],
                        help='Type of the model.')
    # optional args
    parser.add_argument('--device', type=int, default=0, choices=range(-1, 5),
                        help='GPU device index to use, -1 refers to CPU.')
    parser.add_argument('--log_steps', type=int, default=1, choices=[1, 2, 5, 10],
                        help='Number of epoches per logging.')
    parser.add_argument('--compress_ratio', type=float, default=0.8,
                        choices=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Compression ratio for sketching modules.')
    parser.add_argument('--sampling', type=bool, default=False, choices=[True, False],
                        help='Whether to sample mini-batches.')
    parser.add_argument('--sampling_type', type=str, default='RW', choices=['RW'],
                        help='Sampling strategies.')
    parser.add_argument('--sampling_ratio', type=float, default=1.0,
                        choices=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                        help='Sampling ratio for mini-batches.')
    parser.add_argument('--num_workers', type=int, default=0, choices=range(0, 9),
                        help='Number of workers in data loaders.')
    parser.add_argument('--num_layers', type=int, default=4, choices=range(2, 6),
                        help='Number of layers of the model.')
    parser.add_argument('--hidden_channels', type=int, default=128, choices=[64, 128, 256],
                        help='Number of hidden channels.')
    parser.add_argument('--batchnorm', type=bool, default=False, choices=[True, False],
                        help='Whether to use batch normalization.')
    parser.add_argument('--dropout', type=float, default=0, choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help='Drop out rate.')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'Sigmoid', 'None'],
                        help='Activation function to use.')
    parser.add_argument('--order', type=int, default=2, choices=range(1, 10),
                        help='Order of (approximated) polynomial activation.')
    parser.add_argument('--top_k', type=int, default=8, choices=range(1, 17),
                        help='Top number of entries per row to preserve in the sketched convolution matrices.')
    parser.add_argument('--sketch_mode', type=str, default='all_same',
                        choices=['all_distinct', 'layer_distinct', 'order_distinct', 'all_same'],
                        help='How are the different sketch modules different with respect to each others.')
    parser.add_argument('--lr', type=float, default=1e-3, choices=[1e-1, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
                        help='Learning rate.')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'],
                        help='Optimization algorithm to use.')
    parser.add_argument('--clip_threshold', type=float, default=0.01, choices=[0.01, 0.02, 0.05, 0.1],
                        help='Grad clipping threshold.')
    parser.add_argument('--num_sketches', type=int, default=2, choices=[1, 2, 3, 4, 5],
                        help='Number of sketches in each experiment.')
    parser.add_argument('--num_epochs', type=int, default=500, choices=[10, 20, 50, 100, 200, 300, 500, 1000],
                        help='Number of epochs in each experiment.')
    parser.add_argument('--runs', type=int, default=3, choices=range(1, 11),
                        help='Number of repeated experiments.')
    args = parser.parse_args()
    # parse args
    assert (args.sampling and args.sampling_ratio < 1) or (not args.sampling and args.sampling_ratio == 1)
    if args.model == 'Poly-Act-GCN':
        args.activation = f'{format_ordinal(args.order)} order (learnable) polynomials'

    # print args
    print('#' * 60)
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print('#' * 60)

    # baseline flag
    run_baselines = args.model in ['Poly-Act-GCN', 'Standard-GCN']

    # device
    device = torch.device(args.device if args.device >= 0 else 'cpu')

    # load dataset
    dataset = PygNodePropPredDataset(name='ogbn-{}'.format(args.dataset),
                                     transform=T.ToSparseTensor(), root="../dataset")
    num_features = dataset[0].num_features

    # dataloader
    if run_baselines:
        train_loader = get_baseline_dataloader(dataset, args.sampling, args.sampling_type,
                                               args.sampling_ratio, args.num_layers, args.num_workers)
        test_loader = get_baseline_test_dataloader(dataset)
    else:
        train_loader = get_dataloader(dataset, args.compress_ratio, args.sampling, args.sampling_type,
                                      args.sampling_ratio, args.num_layers, args.order, args.top_k,
                                      args.sketch_mode, args.num_sketches, args.num_workers)
        test_loader = get_test_dataloader(dataset)

    # load model
    if args.model == 'Sketch-GCN':
        model = SketchGCN(num_features, args.hidden_channels, dataset.num_classes,
                          args.num_layers, args.batchnorm, args.dropout, args.order)
    elif args.model == 'Poly-Act-GCN':
        model = PolyActGCN(num_features, args.hidden_channels, dataset.num_classes,
                           args.num_layers, args.batchnorm, args.dropout, args.order)
    elif args.model == 'Standard-GCN':
        model = StandardGCN(num_features, args.hidden_channels, dataset.num_classes,
                            args.num_layers, args.batchnorm, args.dropout, args.activation)
    else:
        raise NotImplementedError

    # evaluator
    evaluator = Evaluator(name='ogbn-{}'.format(args.dataset))

    # logger
    logger = Logger(args.runs, args)

    # optimizer
    model.reset_parameters()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    # load model
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated(device) == 0
    model = model.to(device)
    model_memory_usage = torch.cuda.memory_allocated(device)

    # train entrance
    for run in range(args.runs):
        # reset model
        model.reset_parameters()

        # load test data
        gc.collect()
        torch.cuda.empty_cache()
        test_data = list(test_loader)[0]
        test_data = load_nested_list(test_data, device)
        test_memory_usage = torch.cuda.memory_allocated(device) - model_memory_usage

        # train loop
        for epoch in range(1, 1 + args.num_epochs):
            # initialize
            load_epoch_time = 0
            train_epoch_time = 0
            test_epoch_time = 0
            train_memory_usage = None
            loss = None
            result = None

            # train loop
            start_time = time.time()
            for epoch_data in train_loader:
                # load
                epoch_data = load_nested_list(epoch_data, device)
                load_epoch_time += time.time() - start_time

                # train
                start_time = time.time()
                gc.collect()
                torch.cuda.empty_cache()
                if not run_baselines:
                    loss = train(model, epoch_data, optimizer, args.clip_threshold, args.num_sketches)
                else:
                    loss = train_baselines(model, epoch_data, optimizer)
                train_memory_usage = torch.cuda.memory_allocated(device) - test_memory_usage - model_memory_usage
                train_epoch_time = time.time() - start_time

                # test
                start_time = time.time()
                if not run_baselines:
                    result = test(model, test_data, evaluator)
                else:
                    result = test_baselines(model, test_data, evaluator)
                test_epoch_time = time.time() - start_time

                # load
                start_time = time.time()

            # test
            if epoch % args.log_steps == 0:
                # log
                logger.add_result(run, result, (load_epoch_time, train_epoch_time, test_epoch_time),
                                  (test_memory_usage, train_memory_usage))
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:04d}, '
                      f'Loss: {loss:012.3f}, '
                      f'Train: {100 * train_acc:05.2f}%, '
                      f'Valid: {100 * valid_acc:05.2f}%, '
                      f'Test: {100 * test_acc:05.2f}%, '
                      f'Load Time: {load_epoch_time:06.4f}s, '
                      f'Train Time: {train_epoch_time:06.4f}s, '
                      f'Test Time: {test_epoch_time:06.4f}s, '
                      f'Test Memory: {test_memory_usage / 1048576.0:07.2f}MB, '
                      f'Train Memory: {train_memory_usage / 1048576.0:07.2f}MB')
            # clear
            del epoch_data
            gc.collect()
            torch.cuda.empty_cache()

        # log
        logger.print_statistics(run)
    logger.print_statistics()


# Command line entrance
if __name__ == "__main__":
    main()
