from dataset.NoduleDataset import *
from config import load_config
 
def main():
    # Initialize
    cfg = load_config()
    
    print("Step1 - Pre-process data for CNN models") 
    lidc_dataset = NoduleDataset("DATA/LIDC_spiculation", load=False)
    lungx_dataset = NoduleDataset("DATA/LUNGx_spiculation", load=False)
    
    print("Step2 - Pre-process data for voxel2mesh") 
    data_obj = cfg.data_obj  
    data_obj_ext = cfg.data_obj_ext

    # Run pre-processing
    data = data_obj.pre_process_dataset(cfg)
    data_wo3 = data_obj.pre_process_dataset_wo3(cfg)
    data_ext = data_obj_ext.pre_process_dataset(cfg)

if __name__ == "__main__": 
    main()
