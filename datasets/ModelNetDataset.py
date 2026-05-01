'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
@DATASETS.register_module()
class ModelNet(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.subset = config.subset

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.shape_ids = {}
        if self.num_category == 10:
            self.shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            self.shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            self.shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            self.shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert self.subset in ['train', 'test']
        self.datapath = self.build_datapath(self.subset)
        print_log('The size of %s data is %d' % (self.subset, len(self.datapath)), logger = 'ModelNet')

        if self.process_data:
            self.list_of_points, self.list_of_labels = self.load_processed_split(self.subset, self.datapath)
        else:
            self.list_of_points = None
            self.list_of_labels = None

    def get_save_path(self, split):
        if self.uniform:
            return os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        return os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

    def build_datapath(self, split):
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in self.shape_ids[split]]
        return [
            (shape_names[i], os.path.join(self.root, shape_names[i], self.shape_ids[split][i]) + '.txt')
            for i in range(len(self.shape_ids[split]))
        ]

    def load_processed_split(self, split, datapath):
        save_path = self.get_save_path(split)
        if not os.path.exists(save_path):
            print_log('Processing data %s (only running in the first time)...' % save_path, logger='ModelNet')
            list_of_points = [None] * len(datapath)
            list_of_labels = [None] * len(datapath)

            for index in tqdm(range(len(datapath)), total=len(datapath)):
                fn = datapath[index]
                cls = self.classes[datapath[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                list_of_points[index] = point_set
                list_of_labels[index] = cls

            with open(save_path, 'wb') as f:
                pickle.dump([list_of_points, list_of_labels], f)
        else:
            print_log('Load processed data from %s...' % save_path, logger='ModelNet')
            with open(save_path, 'rb') as f:
                list_of_points, list_of_labels = pickle.load(f)
        return list_of_points, list_of_labels

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]


    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label)
