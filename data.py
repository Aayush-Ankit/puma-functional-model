#import os
#import torch
#import cPickle as pickle
#import numpy
#import torchvision.transforms as transforms
#
#class dataset():
#    def __init__(self, root=None, train=True):
#        self.root = root
#        self.train = train
#        self.transform = transforms.ToTensor()
#        if self.train:
#            train_data_path = os.path.join(root, 'train_data')
#            train_labels_path = os.path.join(root, 'train_labels')
#            self.train_data = numpy.load(open(train_data_path, 'r'))
#            self.train_data = torch.from_numpy(self.train_data.astype('float32'))
#            self.train_labels = numpy.load(open(train_labels_path, 'r')).astype('int')
#        else:
#            test_data_path = os.path.join(root, 'test_data')
#            test_labels_path = os.path.join(root, 'test_labels')
#            self.test_data = numpy.load(open(test_data_path, 'r'))
#            self.test_data = torch.from_numpy(self.test_data.astype('float32'))
#            self.test_labels = numpy.load(open(test_labels_path, 'r')).astype('int')
#
#    def __len__(self):
#        if self.train:
#            return len(self.train_data)
#        else:
#            return len(self.test_data)
#
#    def __getitem__(self, index):
#        if self.train:
#            img, target = self.train_data[index], self.train_labels[index]
#        else:
#            img, target = self.test_data[index], self.test_labels[index]
#
#
#        return img, target

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

_DATASETS_MAIN_PATH = './Datasets'
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
    }
}


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name == 'cifar10':
        return datasets.CIFAR10(root=_dataset_path['cifar10'],
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=_dataset_path['cifar100'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        return datasets.ImageFolder(root=path,
                                    transform=transform,
                                    target_transform=target_transform)

