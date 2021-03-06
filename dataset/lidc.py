import numpy as np

import glob
import pickle

import torch
import pandas as pd
from tqdm import tqdm

from dataset.data import get_item, sample_to_sample_plus
from utils.metrics import jaccard_index

from external.voxel2mesh.utils.utils_common import DataModes
from external.voxel2mesh.utils.metrics import chamfer_weighted_symmetric

MODALITIES = ['CT', 'ard', 'spikes', 'nodule']

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

class Sample:
    def __init__(self, x, y, atlas=None):
        self.x = x
        self.y = y
        self.atlas = atlas

class SamplePlus:
    def __init__(self, x, y, y_outer=None, x_super_res=None, y_super_res=None, y_outer_super_res=None, shape=None):
        self.x = x
        self.y = y
        self.y_outer = y_outer
        self.x_super_res = x_super_res
        self.y_super_res = y_super_res  
        self.shape = shape

  
class LIDCDataset():
    def __init__(self, data, pids, nids, metadata, cfg, mode, name="LIDC"): 
        self.data = data  
        self.pids = pids
        self.nids = nids
        self.metadata = metadata

        self.cfg = cfg
        self.mode = mode
        self.name = name
 

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx] 
        #print(self.pids[idx])
        item = get_item(item, self.mode, self.cfg) 
        item['pid'] = self.pids[idx].upper()
        item['nid'] = int(self.nids[idx])
        self.metadata.loc[:, "PID"] = self.metadata.PID.apply(lambda x : x.upper())

        item['metadata'] = self.metadata.query(f"PID=='{item['pid']}'").iloc[item['nid']-1].to_dict()
    
        return item

  

class LIDC():
    def pick_surface_points(self, y_outer, point_count):
        idxs = torch.nonzero(y_outer) 
        perm = torch.randperm(len(idxs))

        y_outer = y_outer * 0  
        idxs = idxs[perm[:point_count]]
        y_outer[idxs[:,0], idxs[:,1], idxs[:,2]] = 1
        return y_outer

    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.patch_shape

        data_root = cfg.dataset_path
        data = {}
        for datamode in [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]:
            print("LIDC", datamode, 'dataset')
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                new_samples, samples, sample_pids, sample_nids, metadata = pickle.load(handle)
                data[datamode] = LIDCDataset(new_samples, sample_pids, sample_nids, metadata, cfg, datamode, "LIDC") 

        return data
    
    def quick_load_data_wo3(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.patch_shape

        data_root = cfg.dataset_path
        data = {}
        for datamode in [DataModes.TRAINING, DataModes.VALIDATION]:
            print("LIDC without class 3", datamode, 'dataset')
            with open(data_root + '/pre_computed_data_{}_{}_wo3.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                new_samples, samples, sample_pids, sample_nids, metadata = pickle.load(handle)
                data[datamode] = LIDCDataset(new_samples, sample_pids, sample_nids, metadata, cfg, datamode, "LIDC_12vs45") 

        return data

    def pre_process_dataset(self, cfg):
        data_root = cfg.dataset_path
        metadata = pd.read_csv(data_root+"../LIDC.csv")
        metadata = metadata.query("NID==1")
        metadata.loc[:, "Malignancy"] = (metadata.Malignancy > 3).values.copy()
        metadata72 = pd.read_csv(data_root+"../LIDC72.csv")
        samples = glob.glob(f"{data_root}LIDC*s_0*CT.npy")
 
        pids = []
        nids = []
        inputs = []
        labels = []

        print('Data pre-processing - LIDC Dataset')
        with tqdm(samples) as pbar:
            for sample in pbar:
                if 'pickle' not in sample:
                    pid = sample.split("/")[-1].split("_")[0]
                    nid = sample.split("/")[-1].split("_")[1]
                    pbar.set_description("Processing %s %s" % (pid, nid))

                    pids += [pid]
                    nids += [nid]
                    x = torch.from_numpy(np.load(sample)[0])
                    inputs += [x]
                    y = torch.from_numpy(np.load(sample.replace("CT.npy", "spikes.npy"))) # spike segmenation with nodule area
                    y1 = torch.from_numpy(np.load(sample.replace("CT.npy", "ard.npy"))[0]) # area distortion map
                    y2 = torch.from_numpy(np.load(sample.replace("CT.npy", "nodule.npy"))[0]) # nodule segmentation
                    y = (2*(y == 2).type(torch.uint8) + (y == 3).type(torch.uint8)) * (y1 <= 0).type(torch.uint8) # spikes - (spiculation=2 + lobulation=1) in negative ard
                    y = 3*y2 - y.type(torch.uint8) # apply nodule mask - 3 nodule, 2 lobulation, 1 spiculation
                    
                    #y[y==1] = 5 # nodule
                    #y[y==2] = 1 # spiculation
                    #y[y==3] = 1 # lobulation
                    #y[y==4] = 2 # attachment
                    #y[y>=5] = 2 # others
                    labels += [y]

        print('\nSaving pre-processed data to disk')
        np.random.seed(34234)
        train_val_idx = [x for x in range(len(inputs)) if pids[x] not in selected]
        test_idx = [x for x in range(len(inputs)) if pids[x] in selected]
        perm = np.random.permutation(train_val_idx) 
        counts = [perm[:len(train_val_idx)//10*8], perm[len(train_val_idx)//10*8:], test_idx]
 
        data = {}
        down_sample_shape = cfg.patch_shape


        for i, datamode in enumerate([DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]):
            samples = []
            sample_pids = []
            sample_nids = []
 
            for j in counts[i]: 
                pid = pids[j]
                nid = nids[j]
                x = inputs[j]
                y = labels[j]

                samples.append(Sample(x, y)) 
                sample_pids.append(pid)
                sample_nids.append(nid)

            if datamode == DataModes.TESTING:
                metadata_ = metadata.loc[metadata.PID.isin(sample_pids)]
                metadata_.loc[:, "Malignancy_weak"] = metadata_.Malignancy
                metadata_.loc[:, "Malignancy"] = metadata72.PMalignancy.values == 2
            else:
                metadata_ = metadata.loc[metadata.PID.isin(sample_pids)]
            print(metadata_)
            new_samples = sample_to_sample_plus(samples, cfg, datamode)
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'wb') as handle:
                pickle.dump((new_samples, samples, sample_pids, sample_nids, metadata_), handle, protocol=pickle.HIGHEST_PROTOCOL)

            data[datamode] = LIDCDataset(samples, sample_pids, sample_nids, metadata_, cfg, datamode, "LIDC")
        
        print('Pre-processing complete') 
        return data
    
    def pre_process_dataset_wo3(self, cfg):
        data_root = cfg.dataset_path
        metadata = pd.read_csv(data_root+"../LIDC.csv")
        metadata = metadata.query("NID==1")
        ambiguos = (metadata.Malignancy == 3) & (~metadata.PID.isin(selected))
        metadata.Malignancy = metadata.Malignancy > 3 # Radiological Malignancy (RM)
        metadata = metadata.loc[~ambiguos]
        metadata72 = pd.read_csv(data_root+"../LIDC72.csv")
        samples = glob.glob(f"{data_root}LIDC*s_0*0.npy")
        samples = [sample for sample in samples if sample.split("/")[-1].split("_")[0] in metadata.PID.values]
        
        pids = []
        nids = []
        inputs = []
        labels = []

        print('Data pre-processing - LIDC Dataset without Malignancy class 3')
        with tqdm(samples) as pbar:
            for sample in pbar:
                if 'pickle' not in sample:
                    pid = sample.split("/")[-1].split("_")[0]
                    nid = sample.split("/")[-1].split("_")[1]
                    pbar.set_description("Processing %s %s" % (pid, nid))

                    pids += [pid]
                    nids += [nid]
                    x = torch.from_numpy(np.load(sample)[0])
                    inputs += [x]
                    y = torch.from_numpy(np.load(sample.replace("0.npy", "seg.npy"))) # spike segmenation with nodule area
                    y1 = torch.from_numpy(np.load(sample.replace("0.npy", "1.npy"))[0]) # area distortion map
                    y2 = torch.from_numpy(np.load(sample.replace("0.npy", "2.npy"))[0]) # nodule segmentation
                    y = (2*(y == 2).type(torch.uint8) + (y == 3).type(torch.uint8)) * (y1 <= 0).type(torch.uint8) # spikes
                    y = 3*y2 - y.type(torch.uint8) # apply nodule mask
                    
                    #y[y==1] = 5 # nodule
                    #y[y==2] = 1 # spiculation
                    #y[y==3] = 1 # lobulation
                    #y[y==4] = 2 # attachment
                    #y[y>=5] = 2 # others
                    labels += [y]

        print('\nSaving pre-processed data to disk')
        np.random.seed(34234)
        train_val_idx = [x for x in range(len(pids)) if pids[x] not in selected]
        test_idx = [x for x in range(len(pids)) if pids[x] in selected]
        perm = np.random.permutation(train_val_idx) 
        counts = [perm[:len(train_val_idx)//10*8], perm[len(train_val_idx)//10*8:], test_idx]
 
        data = {}
        down_sample_shape = cfg.patch_shape


        for i, datamode in enumerate([DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]):
            samples = []
            sample_pids = []
            sample_nids = []
 
            for j in counts[i]: 
                pid = pids[j]
                nid = nids[j]
                x = inputs[j]
                y = labels[j]

                samples.append(Sample(x, y)) 
                sample_pids.append(pid)
                sample_nids.append(nid)

            if datamode == DataModes.TESTING:
                metadata_ = metadata.loc[metadata.PID.isin(sample_pids)]
                metadata_.loc[:, "Malignancy_weak"] = metadata_.Malignancy # Radiological Malignancy (RM)
                metadata_.loc[:, "Malignancy"] = metadata72.PMalignancy.values > 1 # Pathological Malignancy (PM)
            else:
                metadata_ = metadata.loc[metadata.PID.isin(sample_pids)]
            print(metadata_)
            new_samples = sample_to_sample_plus(samples, cfg, datamode)
            with open(data_root + '/pre_computed_data_{}_{}_wo3.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'wb') as handle:
                pickle.dump((new_samples, samples, sample_pids, sample_nids, metadata_), handle, protocol=pickle.HIGHEST_PROTOCOL)

            data[datamode] = LIDCDataset(samples, sample_pids, sample_nids, metadata_, cfg, datamode)
        
        print('Pre-processing complete') 
        return data
 
    def evaluate(self, target, pred, cfg):
        results = {}

        if target.voxel is not None: 
            val_jaccard = jaccard_index(target.voxel, pred.voxel, cfg.num_classes)
            results['jaccard'] = val_jaccard

        if target.mesh is not None:
            target_points = target.points
            pred_points = pred.mesh
            val_chamfer_weighted_symmetric = np.zeros(len(target_points))

            for i in range(len(target_points)):
                if target_points[i].size()[1] == 0:
                    val_chamfer_weighted_symmetric[i] = 0
                else:
                    val_chamfer_weighted_symmetric[i] = chamfer_weighted_symmetric(target_points[i], pred_points[i]['vertices'])


            results['chamfer_weighted_symmetric'] = val_chamfer_weighted_symmetric

        return results

    def update_checkpoint(self, best_so_far, new_value):

        key = 'jaccard'
        new_value = new_value[DataModes.VALIDATION][key]

        if best_so_far is None:
            return True
        else:
            best_so_far = best_so_far[DataModes.VALIDATION][key]
            return True if np.nanmean(new_value) > np.nanmean(best_so_far) else False



 
