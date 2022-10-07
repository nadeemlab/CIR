import sys
import numpy as np
import glob
import torch
import pickle
from tqdm import tqdm

from dataset.lidc import *

from dataset.data import sample_to_sample_plus
from external.voxel2mesh.utils.utils_common import DataModes

class MSKLung(LIDC):
    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.patch_shape

        data_root = cfg.ext_dataset1_path
        data = {}
        for datamode in [DataModes.TRAINING, DataModes.TESTING]:
            print("MSKLung", datamode, 'dataset')
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                new_samples, samples, sample_pids, sample_nids, metadata = pickle.load(handle)
                data[datamode] = LIDCDataset(new_samples, sample_pids, sample_nids, metadata, cfg, datamode, "MSKLung") 

        return data

    def pre_process_dataset(self, cfg):
        data_root = cfg.ext_dataset1_path
        samples_train = glob.glob(f"{data_root}*_1-*CT.npy")
        samples_test = glob.glob(f"{data_root}*_2-*CT.npy")
 
        pids = []
        nids = []
        inputs = []
        labels = []

        print('Data pre-processing - MSKLung Dataset')
        with tqdm(samples_train+samples_test) as pbar:
            for sample in pbar:
                if 'pickle' not in sample:
                    pid = sample.split("/")[-1].split("_")[0]
                    nid = sample.split("/")[-1].split("_")[1]
                    sid = sample.split("/")[-1].split("_")[-2]
                    pbar.set_description("Processing %s %s %s" % (pid, nid, sid))
                    
                    pids += [pid]
                    nids += [nid]
                    x = torch.from_numpy(np.load(sample)[0])
                    inputs += [x]
                    y = torch.from_numpy(np.load(sample.replace("CT.npy", "spikes.npy"))) # spike segmenation with nodule area
                    y1 = torch.from_numpy(np.load(sample.replace("CT.npy", "ard.npy"))[0]) # area distortion map
                    y2 = torch.from_numpy(np.load(sample.replace("CT.npy", "nodule.npy"))[0]) # nodule segmentation
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
        n = len(samples_train)
        m = len(samples_test)
        counts = [range(n), range(n,n+m)]
 
        data = {}
        down_sample_shape = cfg.patch_shape

        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING]):
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
            '''
            metadata = pd.read_csv(data_root + "../MSKLung.csv")
            if datamode == DataModes.TRAINING:
                metadata = metadata.iloc[0:10]
            else:
                metadata = metadata.iloc[10:]
            metadata.loc[:, "Malignancy"] = metadata.PMalignancy > 1 # Pathological Malignancy (PM)
            print(metadata)
            '''
            metadata = None
            new_samples = sample_to_sample_plus(samples, cfg, datamode)
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'wb') as handle:
                pickle.dump((new_samples, samples, sample_pids, sample_nids, metadata), handle, protocol=pickle.HIGHEST_PROTOCOL)

            data[datamode] = LIDCDataset(samples, sample_pids, sample_nids, metadata, cfg, datamode, "LUNGx")
        
        print('Pre-processing complete') 
        return data
 


 
