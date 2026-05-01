import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .build import DATASETS
from utils.logger import print_log


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_radius = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if max_radius > 0:
        pc = pc / max_radius
    return pc


@DATASETS.register_module()
class ShapeNetPart(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_classes = getattr(config, 'NUM_CLASSES', 16)
        self.num_parts = getattr(config, 'NUM_PARTS', 50)
        self.subset = config.subset

        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        with open(self.catfile, 'r') as file_obj:
            for line in file_obj:
                category_name, category_id = line.strip().split()
                self.cat[category_name] = category_id

        self.classes = dict(zip(self.cat.keys(), range(len(self.cat))))
        self.class_idx_to_cat = {index: category for category, index in self.classes.items()}
        self.seg_classes = {
            'Earphone': [16, 17, 18],
            'Motorbike': [30, 31, 32, 33, 34, 35],
            'Rocket': [41, 42, 43],
            'Car': [8, 9, 10, 11],
            'Laptop': [28, 29],
            'Cap': [6, 7],
            'Skateboard': [44, 45, 46],
            'Mug': [36, 37],
            'Guitar': [19, 20, 21],
            'Bag': [4, 5],
            'Lamp': [24, 25, 26, 27],
            'Table': [47, 48, 49],
            'Airplane': [0, 1, 2, 3],
            'Pistol': [38, 39, 40],
            'Chair': [12, 13, 14, 15],
            'Knife': [22, 23],
        }
        self.seg_label_to_cat = {}
        for category_name, labels in self.seg_classes.items():
            for label in labels:
                self.seg_label_to_cat[label] = category_name

        split_dir = os.path.join(self.root, 'train_test_split')
        with open(os.path.join(split_dir, 'shuffled_train_file_list.json'), 'r') as file_obj:
            train_ids = set(item.split('/')[2] for item in json.load(file_obj))
        with open(os.path.join(split_dir, 'shuffled_val_file_list.json'), 'r') as file_obj:
            val_ids = set(item.split('/')[2] for item in json.load(file_obj))
        with open(os.path.join(split_dir, 'shuffled_test_file_list.json'), 'r') as file_obj:
            test_ids = set(item.split('/')[2] for item in json.load(file_obj))

        split_filter = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'trainval': train_ids | val_ids,
        }
        if self.subset not in split_filter:
            raise ValueError(f'Unsupported ShapeNetPart subset: {self.subset}')

        self.datapath = []
        target_ids = split_filter[self.subset]
        for category_name, category_id in self.cat.items():
            category_dir = os.path.join(self.root, category_id)
            filenames = sorted(os.listdir(category_dir))
            for filename in filenames:
                token = os.path.splitext(filename)[0]
                if token in target_ids:
                    self.datapath.append(
                        (
                            category_name,
                            token,
                            os.path.join(category_dir, f'{token}.txt'),
                        )
                    )

        self.cache = {}
        self.cache_size = 20000
        print_log(
            'The size of %s data is %d' % (self.subset, len(self.datapath)),
            logger='ShapeNetPart',
        )

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, class_label, seg_label = self.cache[index]
        else:
            category_name, _, filepath = self.datapath[index]
            class_label = np.array(self.classes[category_name], dtype=np.int64)
            data = np.loadtxt(filepath).astype(np.float32)
            point_set = data[:, 0:6] if self.use_normals else data[:, 0:3]
            seg_label = data[:, -1].astype(np.int64)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, class_label, seg_label)

        point_set = point_set.copy()
        seg_label = seg_label.copy()

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        choice = np.random.choice(len(seg_label), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg_label = seg_label[choice]

        _, token, _ = self.datapath[index]
        points_tensor = torch.from_numpy(point_set).float()
        class_tensor = torch.tensor(int(class_label), dtype=torch.long)
        seg_tensor = torch.from_numpy(seg_label).long()
        return 'ShapeNetPart', token, (points_tensor, class_tensor, seg_tensor)
