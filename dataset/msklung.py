import sys
import numpy as np
import glob
import torch
import pickle
from tqdm import tqdm

from dataset.lidc import *

from dataset.data import sample_to_sample_plus
from external.voxel2mesh.utils.utils_common import DataModes
import pandas as pd
from sklearn.model_selection import train_test_split

class MSKLung(LIDC):
    def quick_load_data(self, cfg, trial_id):
        # assert cfg.patch_shape == (64, 256, 256), 'Not supported'
        down_sample_shape = cfg.patch_shape

        data_root = cfg.ext_dataset1_path
        data = {}
        for datamode in [DataModes.TRAINING, DataModes.VALIDATION, DataModes.TESTING]:
            print("MSKLung", datamode, 'dataset')
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'rb') as handle:
                new_samples, samples, sample_pids, sample_nids, metadata = pickle.load(handle)
                data[datamode] = LIDCDataset(new_samples, sample_pids, sample_nids, metadata, cfg, datamode, "MSKLung") 

        return data
    
    def train_valid_test_split(self, data_root):
      """
      Prepare stratified train/valid/test (60/20/20) split on synapse/msk common datasets using Synapse Genomics/Clinical
      spreadsheet.

      Args:
        data_root (TYPE): data root which also includes the Genomic/Clinical spreadsheet.

      Returns:
        None.

      """
      
      genomics_file = f"{data_root}../MSK_Clinic_Genomic.csv"
      df = pd.read_csv(genomics_file)
      
      train_valid, test = train_test_split(df, test_size=0.2, random_state=0, stratify=df[['label']])
      train, valid = train_test_split(train_valid, test_size=0.2, random_state=0, stratify=train_valid[['label']])
      
      train_pids = train['radiology_accession_number'].to_list()
      test_pids = test['radiology_accession_number'].to_list()
      valid_pids = valid['radiology_accession_number'].to_list()
      
      # train_pids = train_pids[:5]
      # test_pids = test_pids[:5]
      # valid_pids = valid_pids[:5]
      
      samples_train = []
      samples_test = []
      samples_valid = []

      for pid in train_pids:
        samples = glob.glob(f"{data_root}*{pid}_1-*CT.npy") + glob.glob(f"{data_root}*{pid}_2-*CT.npy") +\
          glob.glob(f"{data_root}*{pid}_3-*CT.npy") + glob.glob(f"{data_root}*{pid}_4-*CT.npy")
    
        if len(samples) > 0:
          samples_train = samples_train + samples
        

      for pid in test_pids:
        samples = glob.glob(f"{data_root}*{pid}_1-*CT.npy") + glob.glob(f"{data_root}*{pid}_2-*CT.npy") +\
          glob.glob(f"{data_root}*{pid}_3-*CT.npy") + glob.glob(f"{data_root}*{pid}_4-*CT.npy")
        if len(samples) > 0:
          samples_test = samples_test + samples
      
      for pid in valid_pids:
        samples = glob.glob(f"{data_root}*{pid}_1-*CT.npy") + glob.glob(f"{data_root}*{pid}_2-*CT.npy") +\
          glob.glob(f"{data_root}*{pid}_3-*CT.npy") + glob.glob(f"{data_root}*{pid}_4-*CT.npy")
        if len(samples) > 0:
          samples_valid = samples_valid + samples
          
      print(len(samples_train), len(samples_test), len(samples_valid))
      return samples_train, samples_test, samples_valid
        
    def pre_process_dataset(self, cfg):
        data_root = cfg.ext_dataset1_path
        
        samples_train, samples_test, samples_valid = self.train_valid_test_split(data_root)
              
        # samples_train = glob.glob(f"{data_root}*_1-*CT.npy")
        # samples_test = glob.glob(f"{data_root}*_2-*CT.npy")
 
        pids = []
        nids = []
        inputs = []
        labels = []

        print('Data pre-processing - MSKLung Dataset')
        with tqdm(samples_train+samples_test+samples_valid) as pbar:
            for sample in pbar:
                if 'pickle' not in sample:
                    pid = sample.split("/")[-1].split("_")[0]
                    nid = sample.split("/")[-1].split("_")[1]
                    sid = sample.split("/")[-1].split("_")[-2]
                    pbar.set_description("Processing %s %s %s" % (pid, nid, sid))
                    
                    pids += [pid]
                    nids += [nid]
                    x = torch.from_numpy(np.load(sample)[0])
                    # print(x.shape)
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
        l = len(samples_valid)
        counts = [range(n), range(n,n+m), range(n+m, n+m+l)]
 
        data = {}
        down_sample_shape = cfg.patch_shape
        
        genomics_file = f"{data_root}../MSK_Clinic_Genomic.csv"
        metadata = pd.read_csv(genomics_file)
        metadata['PID'] = metadata['radiology_accession_number']
        metadata['PID']=metadata['PID'].values.astype(str)
        
        for i, datamode in enumerate([DataModes.TRAINING, DataModes.TESTING, DataModes.VALIDATION]):
            samples = []
            sample_pids = []
            sample_nids = []
 
            for j in counts[i]: 
                pid = pids[j]
                nid = nids[j]
                x = inputs[j].float()
                y = labels[j].float()
                # print(x.dtype, y.dtype)
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
            # metadata = None
            new_samples = sample_to_sample_plus(samples, cfg, datamode)
            with open(data_root + '/pre_computed_data_{}_{}.pickle'.format(datamode, "_".join(map(str, down_sample_shape))), 'wb') as handle:
                pickle.dump((new_samples, samples, sample_pids, sample_nids, metadata), handle, protocol=pickle.HIGHEST_PROTOCOL)

            data[datamode] = LIDCDataset(samples, sample_pids, sample_nids, metadata, cfg, datamode, "LUNGx")
        
        print('Pre-processing complete') 
        return data
 


 
