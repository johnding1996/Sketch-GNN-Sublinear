import torch
import numpy as np
import math
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from ogb.linkproppred.dataset_pyg import PygLinkPropPredDataset
from torch.utils.data import Dataset, IterableDataset, DataLoader


def get_baseline_dataloader(pyg_dataset, sampling, sampling_type, sampling_ratio, num_layers, num_workers):
    if not sampling:
        dataset = FullGraphDataset(pyg_dataset)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        return dataloader
    else:
        if sampling_type == 'RW':
            dataset = SAINTRWSamplingDataset(pyg_dataset, sampling_ratio=sampling_ratio, num_steps=num_layers)
            dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=collate_fn)
            return dataloader
        else:
            raise NotImplementedError


def get_baseline_test_dataloader(pyg_dataset):
    dataset = BaselineTestDataset(pyg_dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


class FullGraphDataset(Dataset):
    def __init__(self, pyg_dataset):
        super(FullGraphDataset, self).__init__()
        assert isinstance(pyg_dataset, PygNodePropPredDataset) or isinstance(pyg_dataset, PygLinkPropPredDataset)
        self.pyg_dataset = pyg_dataset
        self.nf_mat = None
        self.conv_mat = None
        self.label = None
        self.train_idx = None
        self.__preprocess__()

    def __preprocess__(self):
        self.nf_mat = self.pyg_dataset[0].x.clone()
        self.conv_mat = self.pyg_dataset[0].adj_t.to_symmetric()
        self.label = self.pyg_dataset[0].y.clone()
        self.train_idx = self.pyg_dataset.get_idx_split()['train'].clone()

        # test on smaller subgraph
        #item = torch.arange(math.ceil(self.pyg_dataset[0].x.size(0) * 0.3), dtype=torch.long)
        #self.nf_mat = self.nf_mat[item, :]
        #self.conv_mat = self.conv_mat.saint_subgraph(item)[0]
        #self.label = self.label[item, :]
        #self.train_idx = torch.from_numpy(np.argwhere(np.isin(item.numpy(), self.train_idx)).flatten()).long()

        self.conv_mat = gcn_norm(self.conv_mat, edge_weight=None,
                                 num_nodes=self.nf_mat.size(0), improved=False,
                                 add_self_loops=True, dtype=torch.float32)
        self.nf_mat = F.batch_norm(self.nf_mat, running_mean=None, running_var=None, training=True)

    def __getitem__(self, item):
        assert item == 0
        return self.nf_mat, self.conv_mat, self.label, self.train_idx

    def __len__(self):
        return 1


class SAINTRWSamplingDataset(IterableDataset):
    def __init__(self, pyg_dataset, sampling_ratio, num_steps):
        assert isinstance(pyg_dataset, PygNodePropPredDataset) or isinstance(pyg_dataset, PygLinkPropPredDataset)
        super(SAINTRWSamplingDataset, self).__init__()
        self.pyg_dataset = pyg_dataset
        self.sampling_ratio = sampling_ratio
        self.num_steps = num_steps
        self.num_samples = math.ceil(self.pyg_dataset[0].x.size(0) * self.sampling_ratio / self.num_steps)
        self.nf_mat = None
        self.conv_mat = None
        self.label = None
        self.train_idx = None
        self.__preprocess__()

    def __preprocess__(self):
        self.nf_mat = self.pyg_dataset[0].x.clone()
        self.conv_mat = gcn_norm(self.pyg_dataset[0].adj_t.to_symmetric(), edge_weight=None,
                                 num_nodes=self.pyg_dataset[0].x.size(0), improved=False,
                                 add_self_loops=True, dtype=torch.float32)
        self.label = self.pyg_dataset[0].y.clone()
        self.train_idx = self.pyg_dataset.get_idx_split()['train'].numpy()

    def __getitem__(self, item):
        assert isinstance(item, torch.Tensor) and item.ndim == 1
        nf_mat = self.nf_mat[item, :]
        conv_mat = self.conv_mat.saint_subgraph(item)[0]
        label = self.label[item, :]
        train_idx = torch.from_numpy(np.argwhere(np.isin(item.numpy(), self.train_idx)).flatten()).long()
        return nf_mat, conv_mat, label, train_idx

    def __sample__(self):
        start_nodes = torch.randint(0, self.pyg_dataset[0].x.size(0), (self.num_samples,), dtype=torch.long)
        sampled_idx = self.conv_mat.random_walk(start_nodes, self.num_steps)
        return sampled_idx.unique()

    def __iter__(self):
        graph_size = self.pyg_dataset[0].x.size(0)
        batch_size = math.ceil(graph_size * self.sampling_ratio)
        num_batches = math.floor(1 / self.sampling_ratio)
        assert batch_size * num_batches >= graph_size
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = num_batches
        else:
            per_worker = math.ceil(num_batches / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, num_batches)
        return (self[self.__sample__()] for _ in range(iter_start, iter_end))


class BaselineTestDataset(Dataset):
    def __init__(self, pyg_dataset):
        super(BaselineTestDataset, self).__init__()
        assert isinstance(pyg_dataset, PygNodePropPredDataset) or isinstance(pyg_dataset, PygLinkPropPredDataset)
        self.pyg_dataset = pyg_dataset
        self.nf_mat = None
        self.conv_mat = None
        self.label = None
        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        self.__preprocess__()

    def __preprocess__(self):
        self.nf_mat = self.pyg_dataset[0].x.clone()
        self.conv_mat = self.pyg_dataset[0].adj_t.to_symmetric()
        self.label = self.pyg_dataset[0].y.clone()
        split_idx = self.pyg_dataset.get_idx_split()
        self.train_idx = split_idx['train'].clone()
        self.valid_idx = split_idx['valid'].clone()
        self.test_idx = split_idx['test'].clone()

        # test on smaller subgraph
        #item = torch.arange(math.ceil(self.pyg_dataset[0].x.size(0) * 0.3), dtype=torch.long)
        #self.nf_mat = self.nf_mat[item, :]
        #self.conv_mat = self.conv_mat.saint_subgraph(item)[0]
        #self.label = self.label[item, :]
        #self.train_idx = torch.from_numpy(np.argwhere(np.isin(item.numpy(), self.train_idx)).flatten()).long()
        #self.valid_idx = torch.from_numpy(np.argwhere(np.isin(item.numpy(), self.valid_idx)).flatten()).long()
        #self.test_idx = torch.from_numpy(np.argwhere(np.isin(item.numpy(), self.test_idx)).flatten()).long()

        self.conv_mat = gcn_norm(self.conv_mat, edge_weight=None,
                                 num_nodes=self.nf_mat.size(0), improved=False,
                                 add_self_loops=True, dtype=torch.float32)
        self.nf_mat = F.batch_norm(self.nf_mat, running_mean=None, running_var=None, training=True)

    def __getitem__(self, item):
        assert item == 0
        return self.nf_mat, self.conv_mat, self.label, self.train_idx, self.valid_idx, self.test_idx

    def __len__(self):
        return 1
