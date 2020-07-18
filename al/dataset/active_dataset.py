import numpy as np
from torch.utils.data import Dataset


class MaskDataset(Dataset):

    def __init__(self, dataset, init_indices=[]):
        super().__init__()
        self.dataset = dataset
        self.indices = init_indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


class ActiveDataset():

    def __init__(self, dataset, n_init=100):
        self.dataset = dataset
        self.masklabeled = np.array([False for i in range(len(dataset))])
        self.update_labeled_list()
        init_list = list(np.random.permutation(np.arange(len(dataset)))[:n_init])
        self.add_to_labeled(init_list)

    def _get_initial_dataset(self):
        raise NotImplementedError

    def get_dataset(self, indices):
        raise NotImplementedError

    def update_labeled_list(self):
        self.labeled = [i for i, labeled in enumerate(self.masklabeled) if labeled]
        self.unlabeled_to_all = {}
        j = 0
        for i, labeled in enumerate(self.masklabeled):
            if not labeled:
                self.unlabeled_to_all[j] = i
                j += 1

    def get_labeled(self):
        return MaskDataset(self.dataset, self.labeled)

    def get_unlabeled(self):
        unlabeled_indices = [i for i, labeled in enumerate(self.masklabeled) if not labeled]
        return MaskDataset(self.dataset, unlabeled_indices)

    def add_to_labeled(self, indices):
        print(indices)
        self.masklabeled[np.array(indices)] = True
        self.update_labeled_list()