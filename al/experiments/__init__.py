from .image_classification import *
from .object_detection import *



def set_up_learner(dataset):
    if dataset == 'mnist':
        return set_up_mnist
    elif dataset == 'cifar':
        return set_up_cifar
    elif dataset == 'pascalvoc_detection':
        return set_up_pascalvoc_detection