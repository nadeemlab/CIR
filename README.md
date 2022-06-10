# IRT
Interpretable Radiomics Toolkit

## Dataset
[Dataset](https://mskcc.ent.box.com/s/w9n5ypo48pq3h5lkxwva5vlnwmxzbc05) - Link disables on March 03, 2022. We will put this data someplace else then.
### LIDC_spiculation
883 cases => train & validation set: 811, test set: 72

### LUNGx_spiculation
70 cases => calibration set: 10 (5 benign & 5 malignant), test set : 60


## Usage


### Data preprocessing
#### Download preprocessed data from (link)
Data from LungCancerScreeningRadiomics  
https://mskcc.box.com/s/t7svvzumdvj8lnsy1wlg6pd1acx3zp3d  
Data Preprocessing step1  
https://mskcc.box.com/s/19twc0iu4po8trk2iz8azkk5s5jvp50n  
Data Preprocessing step2  
https://mskcc.box.com/s/q9krcoja4amyjv16kk9ew0z7lklq0zd6  


`DATA/LIDC_spiculation`  
`DATA/LUNGx_spiculation`


#### Lung nodule spiculation data generation using [LungCancerScreeninigRaiomics](https://github.com/taznux/LungCancerScreeningRadiomics) for LIDC-IDRI and LUNGx
run all the steps in the above repo.


#### Run data preprocessiong
```bash
python data_preprocess.py
```

### Run model traninig
```bash
python main.py
```

### Run model test
```bash
python test.py
```

### Run MedicalZooPytorch model training and test
```bash
python train_3d_seg.py
python test_3d_seg.py
```

[the original voxel2mesh](whttps://github.com/cvlab-epfl/)

## Pipeline
### 1. Nodule Detection
Using existing model or Assuming detected by radiologist
1. Input: CT (512x512xN, Anisotropic or isotropic)  
2. Output: Detected Nodule Patch (64x64xM)

### 2. Nodule Segmentation
 1. Input: Nodule Patch (64x64x64, Isotropic) 
 2. Output
    1. Nodule Voxel Mask (64x64x64): UNet and raterized voxel from mesh
    2. Nodule 3D Mesh (Sphere and Nodule): Area Distortion Map

### 3. Spiculation Quantification
 1. Input: Area Distortion Map on the Mesh
 2. Output: Spiculation Features


### 4. Nodule Classification
 1. Input: Deep features, Spiculation features, etc.  
 2. Output: Benign or Malignant
