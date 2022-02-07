# IRT
Interpretable Radiomics Toolkit

## Dataset
[Dataset](https://mskcc.ent.box.com/s/w9n5ypo48pq3h5lkxwva5vlnwmxzbc05)
### LIDC_spiculation
883 cases => train & validation set: 811, test set: 72

### LUNGx_spiculation
70 cases => calibration set: 10 (5 benign & 5 malignant), test set : 60

### Data preprocessing
```bash
# IRT repository
python train_3d_seg.py --no-loadData # this will start 3D segmentation training after data preprocessing
cd external/voxel2mesh
# voxel2mesh - https://github.com/taznux/voxel2mesh
python data_preprocess.py # this only convert LIDC now
```

## Pipeline
### 1. Nodule Detection
Using existing model or Assuming detected by radiologist
1. Input: CT (512x512xN, Anisotropic or isotropic)  
2. Output: Detected Nodule Patch (64x64xM)


### 2. Nodule Segmentation
Using [voxel2mesh](https://github.com/taznux/voxel2mesh) (UNet+mesh decoder)
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


## TODO
1. [ ] Nodule Detection
    1. [ ] Create an interface to connect existing detectors including [pylidc](https://pylidc.github.io/) and lungx database
2. [ ] Nodule Segmentation
   1. [ ] Angle Preserve Deformation - add a loss component
      1. [ ] CPU computation for testing
      2. [ ] GPU computation using [PyTorch3D](https://pytorch3d.org/)
   2. [ ] Improve code quality - [MeshRCNN](https://github.com/facebookresearch/meshrcnn)
3. [ ] Spiculation Quantification 
   1. [ ] Recursive searching algorithm in matlab -> ?

4. [ ] Nodule Classification
   1. [ ] A simple classifier using two FC layer
   2. [ ] ?

