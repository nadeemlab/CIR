# IRT
Interpretable Radiomics Toolkit

## Dataset
[Dataset](https://mskcc.ent.box.com/s/w9n5ypo48pq3h5lkxwva5vlnwmxzbc05) - Link disables on March 03, 2022. We will put this data someplace else then.
### LIDC_spiculation
883 cases => train & validation set: 811, test set: 72

### LUNGx_spiculation
70 cases => calibration set: 10 (5 benign & 5 malignant), test set : 60

### Data preprocessing
#### Download preprocessed data from (link)
`DATA/LIDC_spiculation`  
`DATA/LUNGx_spiculation`


#### Lung nodule spiculation data generation using [LungCancerScreeninigRaiomics](https://github.com/taznux/LungCancerScreeningRadiomics) for LIDC-IDRI and LUNGx
run all the steps in the above repo.

#### Run data preprocessiong
```bash
# IRT repository
python data_preprocess.py

# voxel2mesh - https://github.com/taznux/voxel2mesh
cd external/voxel2mesh
python data_preprocess.py
```

## Pipeline
### 1. Nodule Detection
Using existing model or Assuming detected by radiologist
1. Input: CT (512x512xN, Anisotropic or isotropic)  
2. Output: Detected Nodule Patch (64x64xM)

### 2. Nodule Segmentation
Using [voxel2mesh](https://github.com/taznux/voxel2mesh) - UNet+mesh decoder and modified for lung nodule spiculation from [the original voxel2mesh](whttps://github.com/cvlab-epfl/)
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
