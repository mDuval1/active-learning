import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

from .active_dataset import ActiveDataset
from ..helpers.constants import DATA_ROOT

class MnistDataset(ActiveDataset):

    def __init__(self, indices, n_init=100):
        self.init_dataset = self._get_initial_dataset()
        super().__init__(self.get_dataset(indices))

    def _get_initial_dataset(self):
        return torchvision.datasets.MNIST(
            root=DATA_ROOT, train=True, transform=transforms.ToTensor(), download=True)

    def get_dataset(self, indices):
        print(type(self.init_dataset.targets[0]))
        print(type(self.init_dataset.data[0]))
        return TensorDataset(
            self.init_dataset.data[indices].float() * 2.0 / 255.0 -1.0,
            self.init_dataset.targets[indices]
        )

    def set_validation_dataset(self, dataset):
        self.val_dataset = dataset

    def get_validation_dataset(self):
        return self.val_dataset