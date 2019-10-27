import os
import torch

import numpy as np

from scipy.linalg import expm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor


DATA_PATH = 'data'


def get_dataset(name, use_lcc=True):
    path = os.path.join(DATA_PATH, name)
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    else:
        raise Exception(f'Dataset not known: {name}.')

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = np.array(dataset.data.edge_index.detach().cpu())
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0]).byte(),
            test_mask=torch.zeros(y_new.size()[0]).byte(),
            val_mask=torch.zeros(y_new.size()[0]).byte()
        )
        dataset.data = data

    return dataset


def get_component(dataset, start=0):
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = np.array(dataset.data.edge_index.detach().cpu())
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset):
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc):
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges, mapper):
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_adj_matrix(dataset, data=None):
    if data is None:
        data = dataset.data
    num_nodes = data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix


def get_rw_matrix(adj_matrix, self_loops=0.0):
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + self_loops * np.eye(num_nodes)
    D_tilde = np.diag(1/A_tilde.sum(axis=1))
    H = A_tilde @ D_tilde
    return H


def get_sym_matrix(adj_matrix, self_loops=1.0):
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + self_loops * np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return H


def get_ppr_matrix(adj_matrix, alpha=0.1, t_matrix='sym', self_loops=1.0):
    num_nodes = adj_matrix.shape[0]
    if t_matrix == 'sym':
        if np.diag(adj_matrix).sum() > 0:
            raise Exception('Adjacency matrix already contains self loops.')
        H = get_sym_matrix(adj_matrix, self_loops)
    elif t_matrix == 'rw':
        H = get_rw_matrix(adj_matrix, self_loops)
    else:
        raise Exception(f'Transition matrix not known: {t_matrix}.')
    ppr_matrix = alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha)*H)
    return ppr_matrix


# see Berberidis et al., 2019 (https://arxiv.org/abs/1804.02081)
def get_heat_matrix(adj_matrix, t=5.0, t_matrix='sym', self_loops=1.0):
    num_nodes = adj_matrix.shape[0]
    if t_matrix == 'sym':
        if np.diag(adj_matrix).sum() > 0:
            raise Exception('Adjacency matrix already contains self loops.')
        H = get_sym_matrix(adj_matrix, self_loops)
    elif t_matrix == 'rw':
        H = get_rw_matrix(adj_matrix, self_loops)
    else:
        raise Exception(f'Transition matrix not known: {t_matrix}.')
    heat_matrix = expm(-t*(np.eye(num_nodes) - H))
    return heat_matrix


def get_top_k_matrix(A, k=16, normalization='col_one'):
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    if normalization == 'None':
        A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    elif normalization == 'col_one':
        A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1 # avoid dividing by zero
        A = A/norm
    elif normalization == 'row_one':
        A = A.transpose()
        A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1 # avoid dividing by zero
        A = A/norm
        A = A.transpose()
    elif normalization == 'col_weights':
        weights = A.sum(axis=0)
        A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1 # avoid dividing by zero
        A = A*weights/norm
    elif normalization == 'row_weights':
        A = A.transpose()
        weights = A.sum(axis=0)
        A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1 # avoid dividing by zero
        A = A*weights/norm
        A = A.transpose()
    else:
        raise Exception(f'Normalization not known: {normalization}.')
    return A


def calculate_eps(A, avg_degree):
    num_nodes = A.shape[0]
    B = A.flatten()
    B.sort()
    if avg_degree*num_nodes > len(B):
        return -np.inf
    return B[-avg_degree*num_nodes]


def get_clipped_matrix(A, eps=0.01, normalization='col_one'):
    # if eps >= 1, interpret as avg_degree
    if eps >= 1:
        eps = calculate_eps(A, eps)
        print(f'Selected new threshold {eps}.')
    num_nodes = A.shape[0]
    if normalization == 'None':
        A[A < eps] = 0.
    elif normalization == 'col_one':
        A[A < eps] = 0.
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1 # avoid dividing by zero
        A = A/norm
    elif normalization == 'row_one':
        A[A < eps] = 0.
        norm = A.sum(axis=1).reshape(num_nodes, 1)
        norm[norm <= 0] = 1 # avoid dividing by zero
        A = A/norm
    elif normalization == 'col_weights':
        weights = A.sum(axis=0)
        A[A < eps] = 0.
        norm = A.sum(axis=0)
        norm[norm <= 0] = 1 # avoid dividing by zero
        A = A*weights/norm
    elif normalization == 'row_weights':
        weights = A.sum(axis=1).reshape(num_nodes, 1)
        A[A < eps] = 0.
        norm = A.sum(axis=1).reshape(num_nodes, 1)
        norm[norm <= 0] = 1 # avoid dividing by zero
        A = A*weights/norm
    elif normalization == 'sym_one':
        A[A < eps] = 0.
        col_norm = np.sqrt(A.sum(axis=0))
        row_norm = np.sqrt(A.sum(axis=1).reshape(num_nodes, 1))
        col_norm[col_norm <= 0] = 1 # avoid dividing by zero
        row_norm[row_norm <= 0] = 1
        A = A/col_norm/row_norm
    elif normalization == 'sym_weights':
        col_weights = A.sum(axis=0)
        row_weights = A.sum(axis=1).reshape(num_nodes, 1)
        A[A < eps] = 0.
        col_norm = np.sqrt(A.sum(axis=0))
        row_norm = np.sqrt(A.sum(axis=1).reshape(num_nodes, 1))
        col_norm[col_norm <= 0] = 1 # avoid dividing by zero
        row_norm[row_norm <= 0] = 1
        A = A*col_weights*row_weights/col_norm/row_norm
    else:
        raise Exception(f'Normalization not known: {normalization}.')
    return A


def set_train_val_test_split(seed, data, num_visible=1500, num_per_class=20, visible_seed=1684992425):
    rnd_state = np.random.RandomState(visible_seed)
    num_nodes = data.y.shape[0]
    visible_idx = rnd_state.choice(num_nodes, num_visible, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in visible_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    num_classes = data.y.max() + 1
    for c in range(num_classes):
        class_idx = visible_idx[np.where(data.y[visible_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in visible_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.uint8)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data


class PPRDataset(InMemoryDataset):
    def __init__(self,
                 name='Cora',
                 use_lcc=True,
                 alpha=0.1,
                 t_matrix='sym',
                 self_loops=1.0,
                 k=16,
                 eps=None,
                 sparse_normalization='col_one'):
        self.name = name
        self.use_lcc = use_lcc
        self.alpha = alpha
        self.t_matrix = t_matrix
        self.self_loops = self_loops
        self.k = k
        self.eps = eps
        self.sparse_normalization = sparse_normalization

        super(PPRDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

        print(f'name={str(self)};num_nodes={self.data.x.shape[0]};num_edges={self.data.edge_index.shape[1]}')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):
        base = get_dataset(name=self.name, use_lcc=self.use_lcc)
        # generate adjacency matrix from sparse representation
        adj_matrix = get_adj_matrix(base)
        # obtain exact PPR matrix
        ppr_matrix = get_ppr_matrix(adj_matrix,
                                    alpha=self.alpha,
                                    t_matrix=self.t_matrix,
                                    self_loops=self.self_loops)

        if self.k:
            if self.eps is not None:
                raise Exception('Both k and eps are not None.')
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=self.k, normalization=self.sparse_normalization)
        elif self.eps:
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=self.eps, normalization=self.sparse_normalization)
        else:
            raise Exception('Both k and eps are None.')

        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(ppr_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(ppr_matrix[i, j])
        edge_index = [edges_i, edges_j]

        data = Data(
            x=base.data.x,
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=base.data.y,
            train_mask=torch.zeros(base.data.train_mask.size()[0]).byte(),
            test_mask=torch.zeros(base.data.test_mask.size()[0]).byte(),
            val_mask=torch.zeros(base.data.val_mask.size()[0]).byte()
        )
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __str__(self):
        if self.k is not None:
            k = format(self.k, 'd')
            eps = 'None'
        else:
            k = 'None'
            eps = format(self.eps, '.8f')
        return (
            f'ppr_{self.name}_use_lcc={self.use_lcc}_alpha={self.alpha:.2f}_t_matrix={self.t_matrix}' +
            f'_self_loops={self.self_loops:.2f}_k={k}_eps={eps}_sparse_normalization={self.sparse_normalization}'
        )


# Use approximate heat matrix as described in Berberidis et al., 2019 (https://arxiv.org/abs/1804.02081)
class HeatDataset(InMemoryDataset):
    def __init__(self,
                 name='Cora',
                 use_lcc=True,
                 t=5.0,
                 t_matrix='sym',
                 self_loops=1.0,
                 k=16,
                 eps=None,
                 sparse_normalization='col_one'):
        self.name = name
        self.use_lcc = use_lcc
        self.t = t
        self.t_matrix = t_matrix
        self.self_loops = self_loops
        self.k = k
        self.eps = eps
        self.sparse_normalization = sparse_normalization

        super(HeatDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

        print(f'name={str(self)};num_nodes={self.data.x.shape[0]};num_edges={self.data.edge_index.shape[1]}')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):
        base = get_dataset(name=self.name, use_lcc=self.use_lcc)
        # generate adjacency matrix from sparse representation
        adj_matrix = get_adj_matrix(base)
        # get heat matrix as described in Berberidis et al., 2019
        heat_matrix = get_heat_matrix(adj_matrix,
                                      t=self.t,
                                      t_matrix=self.t_matrix,
                                      self_loops=self.self_loops)

        if self.k:
            if self.eps is not None:
                raise Exception('Both k and eps are not None.')
            heat_matrix = get_top_k_matrix(heat_matrix, k=self.k, normalization=self.sparse_normalization)
        elif self.eps:
            heat_matrix = get_clipped_matrix(heat_matrix, eps=self.eps, normalization=self.sparse_normalization)
        else:
            raise Exception('Both k and eps are None.')

        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(heat_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(heat_matrix[i, j])
        edge_index = [edges_i, edges_j]

        data = Data(x=base.data.x,
                    edge_index=torch.LongTensor(edge_index),
                    edge_attr=torch.FloatTensor(edge_attr),
                    y=base.data.y,
                    train_mask=torch.zeros(base.data.train_mask.size()[0]).byte(),
                    test_mask=torch.zeros(base.data.test_mask.size()[0]).byte(),
                    val_mask=torch.zeros(base.data.val_mask.size()[0]).byte())
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __str__(self):
        if self.k is not None:
            k = format(self.k, 'd')
            eps = 'None'
        else:
            k = 'None'
            eps = format(self.eps, '.8f')
        return (
            f'heat_{self.name}_use_lcc={self.use_lcc}_t={self.t:.2f}_t_matrix={self.t_matrix}' +
            f'_self_loops={self.self_loops:.2f}_k={k}_eps={eps}_sparse_normalization={self.sparse_normalization}'
        )