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
from lib.utils.general import prepare_input
from lib.losses3D import DiceLoss
import lib.visual3D_temp.BaseWriter
from lib.visual3D_temp.BaseWriter import TensorboardWriter

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


def main():
    gpu = 0
    args = get_arguments()

    if args.resume == "":
        print("Please set `--resume`")
        exit(-1)

    utils.reproducibility(args, seed)

    n_gpus = args.n_gpus
    epochs = args.nEpochs
    batch_size = int(args.batchSz / n_gpus)
    num_worker = int(args.num_worker / n_gpus)

    orig_model, optimizer = medzoo.create_model(args)
    orig_model.restore_checkpoint(args.resume)

    criterion = DiceLoss(classes=args.classes, weight=torch.tensor([0.05,1]).cuda(gpu)) # ,skip_index_after=2,weight=torch.tensor([0.00001,1,1,1]).cuda())

    torch.cuda.set_device(gpu)
    model = orig_model.cuda(gpu)

    lidc_dataset = NoduleDataset("/home/wxc151/data/spiculation/LIDC_spiculation", load=args.loadData)
    lidc_72_dataset = NoduleDataset("/home/wxc151/data/spiculation/LIDC_spiculation", load=True)
    lungx_dataset = NoduleDataset("/home/wxc151/data/spiculation/LUNGx_spiculation", load=args.loadData)

    lidc_72_dataset.list = [x for x in lidc_72_dataset.list if x[0].split("/")[-1].split("_")[0] in selected]
    lidc_dataset.list = list(set(lidc_dataset.list) - set(lidc_72_dataset.list))
    test_dataset = lidc_72_dataset
    ext_test_dataset = lungx_dataset

    train_size = int(0.5 * len(lidc_dataset))
    test_size = len(lidc_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(lidc_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    ext_test_loader = DataLoader(ext_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    print("START Evaluation...")
    for name, data_loader in dict(train=train_loader, valid=valid_loader, test=test_loader, ext_test=ext_test_loader).items():
        print(f"\n {name.capitalize()} set (N={len(data_loader.dataset)}): ", end="")
        test_model(args, model, criterion, data_loader)

    # wandb.finish()
    


def test_model(args, model, criterion, data_loader):
    writer = TensorboardWriter(args)
    model.eval()
    for batch_idx, input_tuple in enumerate(data_loader):
        with torch.no_grad():
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=args)
            input_tensor.requires_grad = False

            output = model(input_tensor)
            loss, per_ch_score = criterion(output, target)

            writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val', batch_idx)

    writer.display_terminal(len(data_loader), 0, mode='val', summary=True)
    val_loss = writer.data['val']['loss'] / writer.data['val']['count']
    print(f"val loss: {val_loss}")
    writer.write_end_of_epoch(0)
    writer.reset('val')



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
    args.save = './evaluations/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(utils.datestr(), args.dataset_name)

    return args


if __name__ == "__main__":
    main()
