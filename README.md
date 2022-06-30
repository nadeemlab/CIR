<!-- PROJECT LOGO -->
<br />
<p align="center">
    <h1 align="center"><strong>Clinically-Interpretable Radiomics</strong></h1>
    <p align="center">
    <a href="">MICCAI'22 Paper</a>
    |
    <a href="https://arxiv.org/pdf/1808.08307.pdf">CMPB'21 Paper</a>
    |
    <a href="https://zenodo.org/record/6762573">CIRDataset</a>
    |
    <a href="https://github.com/choilab-jefferson/LungCancerScreeningRadiomics">Annotation Pipeline</a>
    |
    <a href="#installation">Installation</a>
    |
    <a href="#usage">Usage</a>
    |
    <a href="#docker">Docker</a>
    |
    <a href="https://github.com/nadeemlab/CIR/issues">Issues</a>
  </p>
</p>


This library serves as a one-stop solution for analyzing datasets using clinically-interpretable radiomics (CIR) in cancer imaging. The primary motivation for this comes from our collaborators in radiology and radiation oncology inquiring about the importance of clinically-reported features in state-of-the-art deep learning malignancy/recurrence/treatment response prediction algorithms. Previous methods have performed such prediction tasks but without robust attribution to any clinically reported/actionable features (see extensive literature on sensitivity of attribution methods to hyperparameters). This motivated us to curate datasets by annotating clinically-reported features at voxel/vertex-level on public datasets (using our CMPB'21 [advanced mathmetical algorithms](https://github.com/choilab-jefferson/LungCancerScreeningRadiomics)) and relating these to prediction tasks (bypassing the “flaky” attribution schemes). With the release of these comprehensively-annotated datasets, we hope that previous malignancy prediction methods can also validate their explanations and provide clinically-actionable insights. We also provide strong end-to-end baselines for extracting these hard-to-compute clinically-reported features and using these in different prediction tasks. 

## CIRDataset: A large-scale Dataset for Clinically-Interpretable lung nodule Radiomics and malignancy prediction [MICCAI'22]

*Spiculations/lobulations, sharp/curved spikes on the surface of lung nodules, are good predictors of lung cancer malignancy and hence, are routinely assessed and reported by radiologists as part of the standardized Lung-RADS clinical scoring criteria. Given the 3D geometry of the nodule and 2D slice-by-slice assessment by radiologists, manual spiculation/lobulation annotation is a tedious task and thus no public datasets exist to date for probing the importance of these clinically-reported features in the SOTA malignancy prediction algorithms. As part of this paper, we release a large-scale Clinically-Interpretable Radiomics Dataset, CIRDataset, containing 956 radiologist QA/QC'ed spiculation/lobulation annotations on segmented lung nodules from two public datasets, LIDC-IDRI (N=883) and LUNGx (N=73). We also present an end-to-end deep learning model based on multi-class Voxel2Mesh extension to segment nodules (while preserving spikes), classify spikes (sharp/spiculation and curved/lobulation), and perform malignancy prediction. Previous methods have performed malignancy prediction for LIDC and LUNGx datasets but without robust attribution to any clinically reported/actionable features (due to known hyperparameter sensitivity issues with general attribution schemes). With the release of this comprehensively-annotated dataset and end-to-end deep learning baseline, we hope that malignancy prediction methods can validate their explanations, benchmark against our baseline, and provide clinically-actionable insights. Dataset, code, pretrained models, and docker containers to reproduce the pipeline as well as the results in the manuscript are available in this repository.*

## Dataset
The first CIR dataset, released [here](https://zenodo.org/record/6762573), contains almost 1000 radiologist QA/QC’ed spiculation/lobulation annotations (computed using our published [LungCancerScreeningRadiomics](https://github.com/choilab-jefferson/LungCancerScreeningRadiomics) library [CMPB'21] and QA/QC'ed by a radiologist) on segmented lung nodules for two public datasets, LIDC (with visual radiologist malignancy RM scores for the entire cohort and pathology-proven malignancy PM labels for a subset) and LUNGx (with pathology-proven size-matched benign/malignant nodules to remove the effect of size on malignancy prediction). 
![overview_image](./images/samples.png)*Clinically-interpretable spiculation/lobulation annotation dataset samples; the first column - input CT image; the second column - overlaid semi-automated/QA/QC'ed contours and superimposed area distortion maps (for quantifying/classifying spikes, computed from spherical parameterization -- see our [LungCancerScreeninigRadiomics Library](https://github.com/choilab-jefferson/LungCancerScreeningRadiomics)); the third column - 3D mesh model with vertex classifications, red: spiculations, blue: lobulations, white: nodule base.*

## End-to-End Deep Learning Nodule Segmentation, Spikes' Classification (Spiculation/Lobulation), and Malignancy Prediction Model

We also release our multi-class [Voxel2Mesh](https://github.com/cvlab-epfl/voxel2mesh) extension to provide a strong benchmark for end-to-end deep learning lung nodule segmentation, spikes’ classification (lobulation/spiculation), and malignancy prediction; Voxel2Mesh is the only published method to our knowledge that preserves sharp spikes during segmentation and hence its use as our base model. With the release of this comprehensively-annotated dataset, we hope that previous malignancy prediction methods can also validate their explanations/attributions and provide clinically-actionable insights. Users can also generate spiculation/lobulation annotations from scratch for LIDC/LUNGx as well as new datasets using our [LungCancerScreeningRadiomics](https://github.com/choilab-jefferson/LungCancerScreeningRadiomics) library [CMPB'21].

![architecure_image](./images/CIR_architecture.png)*Depiction of end-to-end deep learning architecture based on multi-class Voxel2Mesh extension. The standard UNet based voxel encoder/decoder (top) extracts features from the input CT volumes while the mesh decoder deforms an initial spherical mesh into increasing finer resolution meshes matching the target shape. The mesh deformation utilizes feature vectors sampled from the voxel decoder through the Learned Neighborhood (LN) Sampling technique and also performs adaptive unpooling with increased vertex counts in high curvature areas. We extend the architecture by introducing extra mesh decoder layers for spiculation and lobulation classification. We also sample vertices (shape features) from the final mesh unpooling layer as input to Fully Connected malignancy prediction network. We optionally add deep voxel-features from the last voxel encoder layer to the malignancy prediction network.*

## Installation
It is highly recommended to install dependencies in either a python virtual environment or anaconda environment. Instructions for python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
(venv) pip install torch==1.11.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
(venv) pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
(venv) pip install wandb sklearn scikit-image ipython ninja pandas opencv-python tqdm
```
Please refer to the this [link](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#3-install-wheels-for-linux) for the details of pytorch3d installation.

## Usage
```bash
git clone --recursive git@github.com:nadeemlab/CIR.git
```
High level usage instructions are detailed below. Detailed instructions at each step, including running pre-trained models, are described in following subsections.

Step 1: Update config.py. You may need to set the path to the dataset and also the directory to save the results. All ready to train/test data is available [here](https://zenodo.org/record/6762573).

Step 2: You have to first perform data pre-processing. `python data_preprocess.py`

Step 3: Now execute `python main.py` and this will start training the network.

Step 4: Test the trained model. `python test.py`

### Data Pre-processing
Pre-processed data will be saved at the dataset directory.

Step 2.0: Generate nrrd files using LungCancerScreeningRadiomics
- Lung nodule spiculation data can be generated from the scratch using [LungCancerScreeninigRadiomics](https://github.com/choilab-jefferson/LungCancerScreeningRadiomics) [CMPB'21] for LIDC-IDRI and LUNGx dataset.  

- Pre-processed data is available [here](https://zenodo.org/record/6762573).
```bash
   tar xjvf CIRDataset_LCSR.tar.bz2
```

Step 2.1: Convert isotropic voxel data from LungCancerScreeningRadiomics to 64x64x64 cubic image patch for 3D CNN models (dataset/NoduleDataset.py)
- Input: Each case consists of four nrrd files (SimpleITK)  
    LIDC-IDRI-0001_CT_1-all.nrrd                - CT Image  
    LIDC-IDRI-0001_CT_1-all-ard.nrrd            - Area Distortion Map  
    LIDC-IDRI-0001_CT_1-all-label.nrrd          - Nodule Segmentation  
    LIDC-IDRI-0001_CT_1-all-spikes-label.nrrd    - Spike Classification - Spiculation:1, Lobulation: 2, Attachment: 3  
- Output: Each case consists of four npy files (numpy) - 64x64x64 cubic image patch  
    LIDC-IDRI-0001_iso0.70_s_0_CT.npy           - CT Image  
    LIDC-IDRI-0001_iso0.70_s_0_ard.npy          - Area Distortion Map  
    LIDC-IDRI-0001_iso0.70_s_0_nodule.npy       - Nodule Segmentation  
    LIDC-IDRI-0001_iso0.70_s_0_spikes.npy        - Spike Classification - Spiculation:1, Lobulation: 2, Attachment: 3  

- Pre-processed data is available [here](https://zenodo.org/record/6762573).
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

- Pre-processed data is available [here](https://zenodo.org/record/6762573).
```bash
   tar xjvf CIRDataset_pickle_for_voxel2mesh.tar.bz2
```

### Running Pre-trained Models
1. Mesh Only model is available [here](https://zenodo.org/record/6762573)
```bash
    tar xjvf pretrained_model-meshonly.tar.bz2
    python test.py --model_path experiments/MICCAI2022/Experiment_001/trial_1
```
2. Mesh+Encoder model is available [here](https://zenodo.org/record/6762573)
```bash
    tar xjvf pretrained_model-mesh+encoder.tar.bz2
    python test.py --model_path experiments/MICCAI2022/Experiment_002/trial_1
```

### Docker
We provide a Dockerfile that can be used to run the models inside a container.
First, you need to install the [Docker Engine](https://docs.docker.com/engine/install/ubuntu/). For using GPU's you also need to install [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). After installing the Docker, you need to follow these steps:

1. Clone this repository.
2. Download data (CIRDataset_pickle_for_voxel2mesh.tar.bz2) available [here](https://zenodo.org/record/6762573).
3. Download pre-trained models, see previous step: Running Pre-trained Models.
4. To create a docker image from the docker file; from top-level repository directory:
```
docker build -f Dockerfile_CIR -t cir_docker .
```
* Note: You may need to modify lines 1 and 9 of Dockerfile_CIR to match your systems' cuda version.
5. Upon successful docker image creation:
```
docker run --gpus all -it cir_docker /bin/bash
```
* Pre-built docker image including data and pre-trained models is available [here](https://hub.docker.com/r/choilab/cir_docker)
```
docker run --gpus all -it choilab/cir_docker /bin/bash
```
6. Then run `python3 test.py --model_path experiments/MICCAI2022/Experiment_001/trial_1` or `python3 test.py --model_path experiments/MICCAI2022/Experiment_002/trial_1` for testing either of the two pre-trained models.

### Reproducibility [MICCAI'22]
The following tables show the expected results of running the pre-trained 'Mesh Only' and 'Mesh+Encoder' models.

*Table1. Nodule (Class0), spiculation (Class1), and lobulation (Class2) peak classification metrics*
<table>
  <tr>
    <th colspan="7">Training</th>
  </tr>
  <tr>
    <td rowspan="2"><b>Network</b></td>
    <td align="center" vetical-align="middel" colspan="3">Chamfer Weighted Symmetric &#8595;</td>
    <td align="center" vetical-align="middel" colspan="3">Jaccard Index &#8593;</td>
  </tr>
  <tr>
    <td>Class0</td>
    <td>Class1</td>
    <td>Class2</td>
    <td>Class0</td>
    <td>Class1</td>
    <td>Class2</td>
  </tr>
  <tr>
    <td><b>Mesh Only</b></td>
    <td>0.009</td>
    <td>0.010</td>
    <td>0.013</td>
    <td>0.507</td>
    <td>0.493</td>
    <td>0.430</td>
  </tr>
  <tr>
    <td><b>Mesh+Encoder</b></td>
    <td>0.008</td>
    <td>0.009</td>
    <td>0.011</td>
    <td>0.488</td>
    <td>0.456</td>
    <td>0.410</td>
  </tr>
  <tr>
    <th colspan="7">Validation</th>
  </tr>
  <tr>
    <td rowspan="2"><b>Network</b></td>
    <td align="center" vetical-align="middel" colspan="3">Chamfer Weighted Symmetric &#8595;</td>
    <td align="center" vetical-align="middel" colspan="3">Jaccard Index &#8593;</td>
  </tr>
  <tr>
    <td>Class0</td>
    <td>Class1</td>
    <td>Class2</td>
    <td>Class0</td>
    <td>Class1</td>
    <td>Class2</td>
  </tr>
  <tr>
    <td><b>Mesh Only</b></td>
    <td>0.010</td>
    <td>0.011</td>
    <td>0.014</td>
    <td>0.526</td>
    <td>0.502</td>
    <td>0.451</td>
  </tr>
  <tr>
    <td><b>Mesh+Encoder</b></td>
    <td>0.014</td>
    <td>0.015</td>
    <td>0.018</td>
    <td>0.488</td>
    <td>0.472</td>
    <td>0.433</td>
  </tr>
  <tr>
    <th colspan="7">Testing LIDC-PM N=72</th>
  </tr>
  <tr>
    <td rowspan="2"><b>Network</b></td>
    <td align="center" vetical-align="middel" colspan="3">Chamfer Weighted Symmetric &#8595;</td>
    <td align="center" vetical-align="middel" colspan="3">Jaccard Index &#8593;</td>
  </tr>
  <tr>
    <td>Class0</td>
    <td>Class1</td>
    <td>Class2</td>
    <td>Class0</td>
    <td>Class1</td>
    <td>Class2</td>
  </tr>
  <tr>
    <td><b>Mesh Only</b></td>
    <td>0.011</td>
    <td>0.011</td>
    <td>0.014</td>
    <td>0.561</td>
    <td>0.553</td>
    <td>0.510</td>
  </tr>
  <tr>
    <td><b>Mesh+Encoder</b></td>
    <td>0.009</td>
    <td>0.010</td>
    <td>0.012</td>
    <td>0.558</td>
    <td>0.541</td>
    <td>0.507</td>
  </tr>
  <tr>
    <th colspan="7">Testing LUNGx N=73</th>
  </tr>
  <tr>
    <td rowspan="2"><b>Network</b></td>
    <td align="center" vetical-align="middel" colspan="3">Chamfer Weighted Symmetric &#8595;</td>
    <td align="center" vetical-align="middel" colspan="3">Jaccard Index &#8593;</td>
  </tr>
  <tr>
    <td>Class0</td>
    <td>Class1</td>
    <td>Class2</td>
    <td>Class0</td>
    <td>Class1</td>
    <td>Class2</td>
  </tr>
  <tr>
    <td><b>Mesh Only</b></td>
    <td>0.029</td>
    <td>0.028</td>
    <td>0.030</td>
    <td>0.502</td>
    <td>0.537</td>
    <td>0.545</td>
  </tr>
  <tr>
    <td><b>Mesh+Encoder</b></td>
    <td>0.017</td>
    <td>0.017</td>
    <td>0.019</td>
    <td>0.506</td>
    <td>0.523</td>
    <td>0.525</td>
  </tr>
</table>
&nbsp;  

*Table 2. Malignancy prediction metrics.*
<table>
  <tr>
    <th colspan="6">Training</th>
  </tr>
  <tr>
    <td><b>Network</b></td>
    <td>AUC</td>
    <td>Accuracy</td>
    <td>Sensitivity</td>
    <td>Specificity</td>
    <td>F1</td>
  </tr>
  <tr>
    <td><b>Mesh Only</b></td>
    <td>0.885</td>
    <td>80.25</td>
    <td>54.84</td>
    <td>93.04</td>
    <td>65.03</td>
  </tr>
  <tr>
    <td><b>Mesh+Encoder</b></td>
    <td>0.899</td>
    <td>80.71</td>
    <td>55.76</td>
    <td>93.27</td>
    <td>65.94</td>
  </tr>
  <tr>
    <th colspan="6">Validation</th>
  </tr>
  <tr>
    <td><b>Network</b></td>
    <td>AUC</td>
    <td>Accuracy</td>
    <td>Sensitivity</td>
    <td>Specificity</td>
    <td>F1</td>
  </tr>
  <tr>
    <td><b>Mesh Only</b></td>
    <td>0.881</td>
    <td>80.37</td>
    <td>53.06</td>
    <td>92.11</td>
    <td>61.90</td>
  </tr>
  <tr>
    <td><b>Mesh+Encoder</b></td>
    <td>0.808</td>
    <td>75.46</td>
    <td>42.86</td>
    <td>89.47</td>
    <td>51.22</td>
  </tr>
  <tr>
    <th colspan="6">Testing LIDC-PM N=72</th>
  </tr>
  <tr>
    <td><b>Network</b></td>
    <td>AUC</td>
    <td>Accuracy</td>
    <td>Sensitivity</td>
    <td>Specificity</td>
    <td>F1</td>
  </tr>
  <tr>
    <td><b>Mesh Only</b></td>
    <td>0.790</td>
    <td>70.83</td>
    <td>56.10</td>
    <td>90.32</td>
    <td>68.66</td>
  </tr>
  <tr>
    <td><b>Mesh+Encoder</b></td>
    <td>0.813</td>
    <td>79.17</td>
    <td>70.73</td>
    <td>90.32</td>
    <td>79.45</td>
  </tr>
  <tr>
    <th colspan="6">Testing LUNGx N=73</th>
  </tr>
  <tr>
    <td><b>Network</b></td>
    <td>AUC</td>
    <td>Accuracy</td>
    <td>Sensitivity</td>
    <td>Specificity</td>
    <td>F1</td>
  </tr>
  <tr>
    <td><b>Mesh Only</b></td>
    <td>0.733</td>
    <td>68.49</td>
    <td>80.56</td>
    <td>56.76</td>
    <td>71.60</td>
  </tr>
  <tr>
    <td><b>Mesh+Encoder</b></td>
    <td>0.743</td>
    <td>65.75</td>
    <td>86.11</td>
    <td>45.95</td>
    <td>71.26</td>
  </tr>
</table>


## Acknowledgments
* This code is inspired by [Voxel2Mesh](https://github.com/cvlab-epfl/voxel2mesh).

## Reference
If you find our work useful in your research or if you use parts of this code or the dataset, please cite the following papers:
```
@article{choi2022cirdataset,
  title={CIRDataset: A large-scale Dataset for Clinically-Interpretable lung nodule Radiomics and malignancy prediction},
  author={Choi, Wookjin and Dahiya, Navdeep and Nadeem, Saad},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2022},
}

@article{choi2021reproducible,
  title={Reproducible and Interpretable Spiculation Quantification for Lung Cancer Screening},
  author={Choi, Wookjin and Nadeem, Saad and Alam, Sadegh R and Deasy, Joseph O and Tannenbaum, Allen and Lu, Wei},
  journal={Computer Methods and Programs in Biomedicine},
  volume={200},
  pages={105839},
  year={2021},
  publisher={Elsevier}
}
```
