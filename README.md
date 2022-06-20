# CIRDataset: A large-scale Dataset and benchmark for Clinically-Interpretable lung nodule Radiomics and malignancy prediction

This repository is a pytorch implementation of end-to-end lung nodule analsys for clincally-interpretable malignancy prediction.

## Installation
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
pip wandb sklearn sckit-image ipython ninja pandas opencv-python tqdm
```
Please refer to the following link for the details of pytorch3d installation.
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#3-install-wheels-for-linux

## Dataset
[Dataset](https://mskcc.ent.box.com/s/w9n5ypo48pq3h5lkxwva5vlnwmxzbc05) - Link disables on March 03, 2022. We will put this data someplace else then.



## Usage
Step 1: Update config.py. You may need to set the path to the dataset and also the directory to save the results.

Step 2: You have to first perform data pre-processing. `python data_preprocess.py`

Step 3: Now execute `python main.py` and this will start training the network.

Step 4: Test the trained model. `python test.py`

### Data preprocessing
Pre-processed data will be save at the dataset directory.

Step 2.0: Generate nrrd files using LungCancerScreeningRadiomics
- Lung nodule spiculation data can be generated from the scratch using  [LungCancerScreeninigRadiomics](https://github.com/taznux/LungCancerScreeningRadiomics) for LIDC-IDRI and LUNGx dataset.  

- Preprocessed data is also available
https://mskcc.box.com/s/t7svvzumdvj8lnsy1wlg6pd1acx3zp3d  
    ```bash
    tar xjvf CIRDataset_LCSR.tar.bz2
    ```

Step 2.1: Convert isotropic voxel data from LungCancerScreeningRadiomics to 64x64x64 cubic image patch for 3D CNN models (dataset/NoduleDataset.py)
- Input: Each case consists of four nrrd files (SimpleITK)
    LIDC-IDRI-0001_CT_1-all.nrrd                - CT Image  
    LIDC-IDRI-0001_CT_1-all-ard.nrrd            - Area Distortion Map  
    LIDC-IDRI-0001_CT_1-all-label.nrrd          - Nodule Segmentation  
    LIDC-IDRI-0001_CT_1-all-peaks-label.nrrd    - Peak Classification - Spiculation:1, Lobulation: 2, Attachment: 3  
- Output: Each case consists of four npy files (numpy) - 64x64x64 cubic image patch
    LIDC-IDRI-0001_iso0.70_s_0_CT.npy           - CT Image  
    LIDC-IDRI-0001_iso0.70_s_0_ard.npy          - Area Distortion Map  
    LIDC-IDRI-0001_iso0.70_s_0_nodule.npy       - Nodule Segmentation  
    LIDC-IDRI-0001_iso0.70_s_0_peaks.npy        - Peak Classification - Spiculation:1, Lobulation: 2, Attachment: 3  

- Preprocessed data is also available
https://mskcc.box.com/s/t7svvzumdvj8lnsy1wlg6pd1acx3zp3d  
    ```bash
    tar xjvf CIRDataset_npy_for_cnn.tar.bz2
    ```
  
Step 2.2: Divide datasets into subsets (Training, Validation, Testing), extract surface voxels, and combine voxel data and outcome data (dataset/lidc.py & dataset/lungx.py)
- Input: Output from the previous step and outcome data
  LIDC.csv - Raiological malignancy (RM) only  
  LIDC72.csv - RM and pathoogical malignancy (PM)  
  LUNGx.csv - PM only  
- Output: pickle files for each subset
  pre_computed_data_trainig_64_64_64.pickle  
  pre_computed_data_validation_64_64_64.pickle (LUNGx does not have this)  
  pre_computed_data_testing_64_64_64.pickle  

- Preprocessed data is also available
https://mskcc.box.com/s/t7svvzumdvj8lnsy1wlg6pd1acx3zp3d  
    ```bash
    tar xjvf CIRDataset_pickle_for_voxel2mesh.tar.bz2
    ```

## Reference
If you find our work useful in your research or if you use parts of this code, please cite our paper:
```
@article{choi2022cirdataset,
  title={CIRDataset: A large-scale Dataset and benchmark for Clinically-Interpretable lung nodule Radiomics and malignancy prediction},
  author={Choi, Wookjin and Dahiya, Navdeep and Nadeem, Saad},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2022},
}
```
