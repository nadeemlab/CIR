# Clinically-Interpretable Radiomics: End-to-End Lung Nodule Segmentation/Classification and Malignancy Prediction

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

#### Download preprocessed data from (link)
Data from LungCancerScreeningRadiomics  
https://mskcc.box.com/s/t7svvzumdvj8lnsy1wlg6pd1acx3zp3d  
Data Preprocessing step1  
https://mskcc.box.com/s/19twc0iu4po8trk2iz8azkk5s5jvp50n  
Data Preprocessing step2  
https://mskcc.box.com/s/q9krcoja4amyjv16kk9ew0z7lklq0zd6  

```bash
mkdir DATA
tar xjvf data.tar.bz2 -C DATA
tar xjvf data_preproc1.tar.bz2 -C DATA
tar xjvf data_preproc2.tar.bz2 -C DATA
```

The preprocessed data will be extracted into `DATA/LIDC_spiculation` and `DATA/LUNGx_spiculation`.


* Lung nodule spiculation data can be generated from the scratch using  [LungCancerScreeninigRaiomics](https://github.com/taznux/LungCancerScreeningRadiomics) for LIDC-IDRI and LUNGx dataset.

