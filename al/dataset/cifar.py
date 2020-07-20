import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Dataset

from .active_dataset import ActiveDataset, MaskDataset
from ..helpers.constants import DATA_ROOT

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


class TransformedDataset(Dataset):

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(x)
        return x, y



class Cifar100Dataset(ActiveDataset):

    def __init__(self, indices, n_init=100, output_dir=None, train=True):
        self.init_dataset = self._get_initial_dataset(train)
        super().__init__(self.get_dataset(indices), n_init=n_init, output_dir=output_dir)

    def _get_initial_dataset(self, train=True):
        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])
        return torchvision.datasets.CIFAR100(
                root=DATA_ROOT, train=train, transform=transform,
                target_transform=None, download=True)

    def get_dataset(self, indices):
        return MaskDataset(self.init_dataset, indices)

    def set_validation_dataset(self, dataset):
        self.val_dataset = dataset

    def get_validation_dataset(self):
        return self.val_dataset