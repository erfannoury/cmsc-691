import os


import numpy as np


class CatsvDogsDataset(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.labels_file = os.path.join(dataset_dir, 'labels.txt')
        assert os.path.exists(self.labels_file)
        self.__read_labels_file__()
        self.train_indices = np.where(self.image_sets == 1)[0]
        self.val_indices = np.where(self.image_sets == 2)[0]
        self.test_indices = np.where(self.image_sets == 3)[0]

    def __read_labels_file__(self):
        self.image_names = []
        self.class_ids = []
        self.image_sets = []
        with open(self.labels_file, 'r') as f:
            for line in f.readlines():
                n, i, s = line.strip().split()
                self.image_names.append(n)
                self.class_ids.append(int(i))
                self.image_sets.append(int(s))
        self.image_names = np.array(self.image_names)
        self.class_ids = np.array(self.class_ids)
        self.image_sets = np.array(self.image_sets)


def read_dataset(dataset_dir):
    return CatsvDogsDataset(dataset_dir)
