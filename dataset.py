from torch.utils.data.dataset import Dataset
from typing import List
import dgl
import torch

class NodeSet(Dataset):
    def __init__(self, node_list: List[int], labels):
        super(NodeSet, self).__init__()
        self.node_list = node_list
        self.labels = labels
        assert len(self.node_list) == len(self.labels)

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        return self.node_list[idx], self.labels[idx]


class NbrSampleCollater(object):
    def __init__(self, graph: dgl.DGLHeteroGraph,
                 block_sampler: dgl.dataloading.BlockSampler):
        self.graph = graph
        self.block_sampler = block_sampler

    def collate(self, batch):
        batch = torch.tensor(batch)
        nodes = batch[:, 0]
        labels = batch[:, 1]
        blocks = self.block_sampler.sample_blocks(self.graph, nodes)
        return blocks, labels
