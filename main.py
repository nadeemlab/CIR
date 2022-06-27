import os
import sys
import logging
import torch
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from utils.train import Trainer
from utils.evaluate import Evaluator
from config import load_config
from model.voxel2mesh_nodule import Voxel2Mesh as network

from external.voxel2mesh.utils.utils_common import DataModes, mkdir


logger = logging.getLogger(__name__)


def init(cfg):

    save_path = cfg.save_path + cfg.save_dir_prefix + \
        str(cfg.experiment_idx).zfill(3)

    os.makedirs(save_path, exist_ok=True)

    trial_id = (len([dir for dir in os.listdir(
        save_path) if 'trial' in dir]) + 1) if cfg.trial_id is None else cfg.trial_id
    trial_save_path = save_path + '/trial_' + str(trial_id)

    if not os.path.isdir(trial_save_path):
        mkdir(trial_save_path)
        #copytree(os.getcwd(), trial_save_path + '/source_code', ignore=ignore_patterns('*.git','*.txt','*.tif', '*.pkl', '*.off', '*.so', '*.json','*.jsonl','*.log','*.patch','*.yaml','wandb','run-*'))

    seed = trial_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation

    return trial_save_path, trial_id


def main():
    # Initialize
    cfg = load_config()
    trial_path, trial_id = init(cfg)

    print('Experiment ID: {}, Trial ID: {}'.format(cfg.experiment_idx, trial_id))

    print("Create network")
    classifier = network(cfg)
    classifier.cuda(cfg.device)

    if cfg.wab:
        wandb.init(name='Experiment_{}/trial_{}'.format(cfg.experiment_idx,
                   trial_id), project="CIR", dir=trial_path)

    print("Initialize optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters(
    )), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    print("Load pre-processed data")
    data_obj = cfg.data_obj
    data = data_obj.quick_load_data(cfg, trial_id)

    loader = DataLoader(data[DataModes.TRAINING],
                        batch_size=classifier.config.batch_size, shuffle=True)

    print("Trainset length: {}".format(loader.__len__()))

    print("Initialize evaluator")
    evaluator = Evaluator(classifier, optimizer, data,
                          trial_path, cfg, data_obj)

    print("Initialize trainer")
    trainer = Trainer(classifier, loader, optimizer,
                      trial_path, evaluator, cfg)

    if cfg.trial_id is not None:
        print("Loading pretrained network")
        save_path = trial_path + '/best_performance/model.pth'
        checkpoint = torch.load(save_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 1

    trainer.train(start_epoch=epoch)


if __name__ == "__main__":
    main()
