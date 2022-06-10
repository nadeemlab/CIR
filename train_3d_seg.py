import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset.NoduleDataset import *
import wandb

sys.path.append("external/MedicalZooPytorch")

import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss
import lib.visual3D_temp.BaseWriter

lib.visual3D_temp.BaseWriter.dict_class_names["LIDC"] = CLASSES

seed = 34234

selected = ['LIDC-IDRI-0072', 'LIDC-IDRI-0090', 'LIDC-IDRI-0138', 'LIDC-IDRI-0149', 'LIDC-IDRI-0162', 'LIDC-IDRI-0163',
            'LIDC-IDRI-0166', 'LIDC-IDRI-0167', 'LIDC-IDRI-0168', 'LIDC-IDRI-0171', 'LIDC-IDRI-0178', 'LIDC-IDRI-0180',
            'LIDC-IDRI-0183', 'LIDC-IDRI-0185', 'LIDC-IDRI-0186', 'LIDC-IDRI-0187', 'LIDC-IDRI-0191', 'LIDC-IDRI-0203',
            'LIDC-IDRI-0211', 'LIDC-IDRI-0212', 'LIDC-IDRI-0233', 'LIDC-IDRI-0234', 'LIDC-IDRI-0242', 'LIDC-IDRI-0246',
            'LIDC-IDRI-0247', 'LIDC-IDRI-0249', 'LIDC-IDRI-0256', 'LIDC-IDRI-0257', 'LIDC-IDRI-0265', 'LIDC-IDRI-0267',
            'LIDC-IDRI-0268', 'LIDC-IDRI-0270', 'LIDC-IDRI-0271', 'LIDC-IDRI-0273', 'LIDC-IDRI-0275', 'LIDC-IDRI-0276',
            'LIDC-IDRI-0277', 'LIDC-IDRI-0283', 'LIDC-IDRI-0286', 'LIDC-IDRI-0289', 'LIDC-IDRI-0290', 'LIDC-IDRI-0314',
            'LIDC-IDRI-0325', 'LIDC-IDRI-0332', 'LIDC-IDRI-0377', 'LIDC-IDRI-0385', 'LIDC-IDRI-0399', 'LIDC-IDRI-0405',
            'LIDC-IDRI-0454', 'LIDC-IDRI-0470', 'LIDC-IDRI-0493', 'LIDC-IDRI-0510', 'LIDC-IDRI-0522', 'LIDC-IDRI-0543',
            'LIDC-IDRI-0559', 'LIDC-IDRI-0562', 'LIDC-IDRI-0568', 'LIDC-IDRI-0580', 'LIDC-IDRI-0610', 'LIDC-IDRI-0624',
            'LIDC-IDRI-0766', 'LIDC-IDRI-0771', 'LIDC-IDRI-0811', 'LIDC-IDRI-0875', 'LIDC-IDRI-0905', 'LIDC-IDRI-0921',
            'LIDC-IDRI-0924', 'LIDC-IDRI-0939', 'LIDC-IDRI-0965', 'LIDC-IDRI-0994', 'LIDC-IDRI-1002', 'LIDC-IDRI-1004']


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=world_size,
            rank=rank)


def cleanup():
    dist.destroy_process_group()

def main():
    args = get_arguments()
    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    if args.n_gpus > 1:
        torch.multiprocessing.spawn(main_worker, nprocs=args.n_gpus, args=(args, ))
    else:
        main_worker(0, args)


def main_worker(gpu, args):
    #if gpu == 0:
    #    wandb.init(project="nodule-segmentation-3D", config=args)
    
    n_gpus = args.n_gpus
    epochs = args.nEpochs
    batch_size = int(args.batchSz / n_gpus)
    num_worker = int(args.num_worker / n_gpus)

    if n_gpus > 1:
        setup(gpu, n_gpus)

    orig_model, optimizer = medzoo.create_model(args)
    criterion = DiceLoss(classes=args.classes, weight=torch.tensor([0.05,1]).cuda(gpu)) # ,skip_index_after=2,weight=torch.tensor([0.00001,1,1,1]).cuda())

    #if gpu == 0:
    #    wandb.watch(orig_model)
    
    torch.cuda.set_device(gpu)
    model = orig_model.cuda(gpu)
    if n_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        dist.barrier()

    
    preprocessing_fn = None
    lidc_dataset = NoduleDataset("DATA/LIDC_spiculation", load=args.loadData)
    lidc_72_dataset = NoduleDataset("DATA/LIDC_spiculation", load=True)
    lungx_dataset = NoduleDataset("DATA/LUNGx_spiculation", load=args.loadData)

    lidc_72_dataset.list = [x for x in lidc_72_dataset.list if x[0].split("/")[-1].split("_")[0] in selected]
    lidc_dataset.list = list(set(lidc_dataset.list) - set(lidc_72_dataset.list))
    test_dataset = lidc_72_dataset
    ext_test_dataset = lungx_dataset

    train_size = int(0.5 * len(lidc_dataset))
    test_size = len(lidc_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(lidc_dataset, [train_size, test_size])

    if n_gpus > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True, sampler=train_sampler)
        #valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True, sampler=valid_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    if gpu == 0:
        trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=train_loader, valid_data_loader=valid_loader)
    else:
        args.save = None
        trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=train_loader)

    print("START TRAINING...")
    trainer.training()

    if n_gpus > 1:
        cleanup()
    #if gpu == 0:
    #    wandb.finish()
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=32)
    parser.add_argument('--dataset_name', type=str, default="LIDC")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--samples_train', type=int, default=1024)
    parser.add_argument('--samples_val', type=int, default=128)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--threshold', default=0.00000000001, type=float)
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--loadData', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='VNET',
                        choices=('UNET3D', 'DENSENET1', "UNET2D", 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
                        "DENSEVOXELNET", 'VNET', 'VNET2', "RESNET3DVAE", "RESNETMED3D", "COVIDNET1", "COVIDNET2", "CNN",
                        "HIGHRESNET"))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='./runs/')

    args = parser.parse_args()

    args.num_worker = 8
    args.save = './saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == "__main__":
    main()
