import os
import argparse
import logging
import torch
import os.path as osp
import numpy as np

import torch.optim as optim
import wandb

from utils.evaluate import Evaluator
from config import load_config
from model.voxel2mesh_nodule import Voxel2Mesh as network

from external.voxel2mesh.utils.utils_common import mkdir


logger = logging.getLogger(__name__)


def init(cfg):
    save_path = cfg.save_path + cfg.save_dir_prefix + \
        str(cfg.experiment_idx).zfill(3)

    trial_id = (len([dir for dir in os.listdir(
        save_path) if 'trial' in dir])) if cfg.trial_id is None else cfg.trial_id
    trial_save_path = save_path + '/trial_' + str(trial_id)

    seed = trial_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation

    return trial_save_path, trial_id


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                        help='path to testing trial (default: none)')
    args = parser.parse_args()

    return args


def main():
    # Initialize
    args = get_arguments()
    cfg = load_config()

    trial_path, trial_id = init(cfg)

    if len(args.model_path) > 0:
        trial_path = args.model_path
        trial_id = int(osp.basename(trial_path).split("_")[-1])
        cfg.experiment_idx = int(osp.basename(
            osp.dirname(trial_path)).split("_")[-1])
    elif trial_id == 0:
        print("There no trial to test")
        exit(-1)

    print('Experiment ID: {}, Trial ID: {}'.format(cfg.experiment_idx, trial_id))

    print("Create network")
    classifier = network(cfg)
    classifier.cuda(cfg.device)
    
    print("Loading pretrained network")
    save_path = trial_path + '/best_performance/model.pth'
    checkpoint = torch.load(save_path)
    epoch = checkpoint['epoch']
    wandb.init(name='Experiment_{}/trial_{}/epoch_{}_test'.format(cfg.experiment_idx,
               trial_id, epoch), project="vm-net", dir=trial_path)
    
    try:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        cfg.deep_features_classifier = not cfg.deep_features_classifier
        classifier = network(cfg)
        classifier.cuda(cfg.device)
        classifier.load_state_dict(checkpoint['model_state_dict'])

    print("Initialize optimizer")
    optimizer = optim.Adam(filter(
        lambda p: p.requires_grad, classifier.parameters()), lr=cfg.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("Load pre-processed data")
    data_obj = cfg.data_obj
    data_obj_ext = cfg.data_obj_ext
    data = data_obj.quick_load_data(cfg, trial_id)
    #data_wo3 = data_obj.quick_load_data_wo3(cfg, trial_id)
    data_ext = data_obj_ext.quick_load_data(cfg, trial_id)

    print("Initialize evaluator")
    evaluator = Evaluator(classifier, optimizer, data,
                          trial_path, cfg, data_obj)
    #evaluator_wo3 = Evaluator(classifier, optimizer, data_wo3, trial_path, cfg, data_obj)
    evaluator_ext = Evaluator(classifier, optimizer,
                              data_ext, trial_path, cfg, data_obj_ext)

    print("LIDC_123vs45")
    evaluator.evaluate_all(epoch)
    # print("LIDC_12vs45")
    # evaluator_wo3.evaluate_all(epoch)
    print("LUNGx")
    evaluator_ext.evaluate_all(epoch)


if __name__ == "__main__":
    main()
