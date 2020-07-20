import os
import logging
import pickle

import numpy as np

import al
from al.dataset import pascal_voc
from al.dataset import active_dataset
from al.model.model_zoo.unet import UNet
from al.model.model_zoo.image_classification import mobilenet
from al.model.ssd import SSDLearner
from al.model.configs import cfg
from al.train.active_train import ActiveTrain
from al.helpers.logger import setup_logger



TRAIN_SIZE = 2000

EXPERIMENT_NAME = 'pascal_voc_segmentation'
OUTPUT_DIR = f'experiments/{EXPERIMENT_NAME}/results'
FIGURE_DIR = f'experiments/{EXPERIMENT_NAME}/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

logger_name = EXPERIMENT_NAME
logger = setup_logger(logger_name, OUTPUT_DIR)
logger.setLevel(logging.DEBUG)

logger.info('Launching simple experiments on Pascal VOC Segmentation')

experiment_parameters = {
    'n_repeats': 2,
    'strategies': ['random_sampling', 'uncertainty_sampling']
}

active_parameters = {
    'assets_per_query': 1000,
    'n_iter': 5,
    'init_size': 10000,
    'compute_score': True,
    'score_on_train': False
}

train_parameters = {
    'batch_size': 4,
    'iterations': 2,
    'learning_rate': 0.001,
    'shuffle': True,
    'momentum': 0.9,
    'weight_decay': 5e-4
}

index_train = np.arange(TRAIN_SIZE)

config_file = 'al/model/configs/unet.yaml'

def get_model_config(config_file):
    cfg.merge_from_file(config_file)
    if 'unet' in config_file:
        model = UNet(cfg)
    cfg.freeze()
    return model, cfg


def set_up():
    logger.info('Setting up datasets...')
    model, cfg = get_model_config(config_file)

    dataset = pascal_voc.PascalVOCSemanticDataset(index_train, n_init=active_parameters['init_size'], cfg=cfg)
    # test_dataset = active_dataset.MaskDataset(dataset._get_initial_dataset(train=False), np.arange(40))
    test_dataset = dataset._get_initial_dataset(train=False)
    dataset.set_validation_dataset(test_dataset)

    logger.info(f'Dataset initial train size : {len(dataset.init_dataset)}')
    logger.info(f'Dataset used train size : {len(dataset.dataset)}')
    logger.info(f'Dataset test size : {len(test_dataset)}')

    logger.info('Setting up models...')

    img, label = dataset.dataset[0]

    print(img.shape, label.shape)
    print(np.unique(label))

    
    learner = SSDLearner(model=model, cfg=cfg, logger_name=logger_name)
    return dataset, learner

logger.info('Launching trainings...')

dataset, learner = set_up()

# strategy='al_for_deep_object_detection'
# strategy='random_sampling'
# trainer = ActiveTrain(learner, dataset, strategy, logger_name)
# scores = trainer.train(train_parameters, **active_parameters)
