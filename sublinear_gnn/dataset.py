import torch
import numpy as np
import math
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from ogb.linkproppred.dataset_pyg import PygLinkPropPredDataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
from initialize_sketch import preprocess_data


def get_dataloader(pyg_dataset, compress_ratio, sampling, sampling_type, sampling_ratio, num_layers, order, top_k,
                   mode, num_sketches, num_workers):
    if not sampling:
        dataset = SketchFullGraphDataset(pyg_dataset, compress_ratio, num_layers, order, top_k, mode, num_sketches)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        return dataloader
    else:
        if sampling_type == 'RW':
            dataset = SketchSAINTRWSamplingDataset(pyg_dataset, sampling_ratio=sampling_ratio, num_steps=num_layers)
            dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=collate_fn)
            return dataloader
        else:
            raise NotImplementedError


def get_test_dataloader(pyg_dataset):
    dataset = TestDataset(pyg_dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


class SketchFullGraphDataset(Dataset):
    def __init__(self, pyg_dataset, compress_ratio, num_layers, order, top_k, mode, num_sketches):
        super(SketchFullGraphDataset, self).__init__()
        assert isinstance(pyg_dataset, PygNodePropPredDataset) or isinstance(pyg_dataset, PygLinkPropPredDataset)
        self.pyg_dataset = pyg_dataset
        self.compress_ratio = compress_ratio
        self.num_layers = num_layers
        self.order = order
        self.top_k = top_k
        self.mode = mode
        self.num_sketches = num_sketches
        self.nf_mat = None
        self.conv_mat = None
        self.label = None
        self.train_idx = None
        self.nf_sketches = []
        self.conv_sketches = []
        self.ll_cs_list = []
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

        for _ in range(self.num_sketches):
            nf_sketches, conv_sketches, ll_cs_list = preprocess_data(
                self.num_layers, in_dim=self.nf_mat.size(0),
                out_dim=math.ceil(self.nf_mat.size(0) * self.compress_ratio), order=self.order, top_k=self.top_k,
                mode=self.mode, nf_mat=self.nf_mat, conv_mat=self.conv_mat,
            )
            self.nf_sketches.append(nf_sketches)
            self.conv_sketches.append(conv_sketches)
            self.ll_cs_list.append(ll_cs_list)

    def __getitem__(self, item):
        assert item == 0
        return self.nf_sketches, self.conv_sketches, self.ll_cs_list, self.label, self.train_idx

    def __len__(self):
        return 1


class SketchSAINTRWSamplingDataset(IterableDataset):
    def __init__(self, pyg_dataset, sampling_ratio, num_steps):
        assert isinstance(pyg_dataset, PygNodePropPredDataset) or isinstance(pyg_dataset, PygLinkPropPredDataset)
        super(SketchSAINTRWSamplingDataset, self).__init__()
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


class TestDataset(Dataset):
    def __init__(self, pyg_dataset):
        super(TestDataset, self).__init__()
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
