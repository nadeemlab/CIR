import os
import glob
import shutil
import pickle
import os.path as osp
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

#           0             1         2              3               4
CLASSES = ['background', 'nodule', 'spiculation', 'lobulation']#, 'attachment']
MODALITIES = ['CT', 'ard', 'nodule', 'peaks']


def crop_img(img_tensor, crop_size, crop):
    if crop_size[0] == 0:
        return img_tensor
    slices_crop, w_crop, h_crop = crop
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_tensor.dim()
    assert inp_img_dim >= 3
    if img_tensor.dim() == 3:
        full_dim1, full_dim2, full_dim3 = img_tensor.shape
    elif img_tensor.dim() == 4:
        _, full_dim1, full_dim2, full_dim3 = img_tensor.shape
        img_tensor = img_tensor[0, ...]

    if full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                     h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    else:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]

    if inp_img_dim == 4:
        return img_tensor.unsqueeze(0)

    return img_tensor


def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor



def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy


def find_non_zero_labels_mask(segmentation_map, th_percent, crop_size, crop):
    d1, d2, d3 = segmentation_map.shape
    segmentation_map[segmentation_map > 0] = 1
    total_voxel_labels = segmentation_map.sum()

    cropped_segm_map = crop_img(segmentation_map, crop_size, crop)
    crop_voxel_labels = cropped_segm_map.sum()

    label_percentage = crop_voxel_labels / total_voxel_labels
    # print(label_percentage,total_voxel_labels,crop_voxel_labels)
    if label_percentage >= th_percent:
        return True
    else:
        return False


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def save_list(name, list):
    with open(name, "wb") as fp:
        pickle.dump(list, fp)


def load_list(name):
    with open(name, "rb") as fp:
        list_file = pickle.load(fp)
    return list_file


def create_sub_volumes(ct_image, ard_image, label, peak_label, samples, crop_size, filename_prefix, th_percent=0.1):
    """
    :param ls: list of modality paths, where the last path is the segmentation map
    :param samples: train/val samples to generate
    :param crop_size: train volume size
    :param sub_vol_path: path for the particular patient
    :param th_percent: the % of the croped dim that corresponds to non-zero labels
    :param crop_type:
    :return:
    """

    ct_np = sitk.GetArrayFromImage(ct_image).astype("float32")
    ard_np = sitk.GetArrayFromImage(ard_image).astype("float32")
    label_np = sitk.GetArrayFromImage(label)#.astype("uint8")
    peak_label_np = sitk.GetArrayFromImage(peak_label)#.astype("uint8")

    images = [
        postproc_medical_image(ct_np, type = 'CT').unsqueeze(0),
        postproc_medical_image(ard_np, type = 'map', crop_size=ct_np.shape).unsqueeze(0), 
        postproc_medical_image(label_np, type = 'label').unsqueeze(0), 
        postproc_medical_image(peak_label_np, type = 'label', crop_size=ct_np.shape),
    ]
    images[3] += images[2][0] # add nodule region other than peaks and attachements
    modalities = len(images)
    list = []
    # print(modalities)
    # print(ls[2])

    full_segmentation_map = images[2][0]
    
    crops = np.nonzero(full_segmentation_map).numpy() - (np.asarray(crop_size)/2).astype(int)
    np.random.shuffle(crops)
    crops = crops.tolist()
    crops.insert(0, ((full_segmentation_map.shape - np.asarray(crop_size))/2).astype(int).tolist())

    #print('Subvolume samples per volume to generate: ', samples)
    for i in range(samples+1):
        while len(crops):
            crop = crops.pop()
            #print(crop)
            if (np.asarray(crop) < 0).sum() == 0 and find_non_zero_labels_mask(full_segmentation_map, th_percent, crop_size, crop):
                #print(full_segmentation_map.shape, th_percent, crop_size, crop)
                tensor_images = [crop_img(images[j], crop_size, crop) for j in range(modalities)]
                break
    
        filename = filename_prefix + '_s_' + str(i) + '_'
        pid = filename_prefix.split("/")[-1].split("_")[0]
        list_saved_paths = []
        for j in range(modalities):
            f_t1 = filename + MODALITIES[j] + '.npy'
            list_saved_paths.append(f_t1)

            np.save(f_t1, tensor_images[j])
            
        list.append(tuple([pid]+list_saved_paths))

    return list


def postproc_medical_image(img_np, type=None, normalization='full_volume_mean', clip_intenisty=True, crop_size=(0, 0, 0), crop=(0, 0, 0),):
    # Intensity outlier clipping
    if clip_intenisty and type == "CT":
        img_np = percentile_clip(img_np)

    # Intensity normalization
    img_tensor = torch.from_numpy(img_np)

    MEAN, STD, MAX, MIN = 0., 1., 1., 0.
    if type == 'CT':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()
    if type == "CT":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))
    img_tensor = crop_img(img_tensor, crop_size, crop)
    return img_tensor


def image_resample(input_image, ref_image=None, iso_voxel_size=1, is_label=False):
    resample_filter = sitk.ResampleImageFilter()

    input_spacing = input_image.GetSpacing()
    input_direction = input_image.GetDirection()
    input_origin = input_image.GetOrigin()
    input_size = input_image.GetSize()

    if ref_image is not None:
        output_spacing = ref_image.GetSpacing()
        output_direction = ref_image.GetDirection()

        flip = ((input_direction[0] > 0) != (output_direction[0] > 0),
            (input_direction[4] > 0) != (output_direction[4] > 0),
            (input_direction[8] > 0) != (output_direction[8] > 0))

        input_image = sitk.Flip(input_image, flip)
        input_image.SetDirection(output_direction)

        input_direction = input_image.GetDirection()
        input_origin = input_image.GetOrigin()

        #print(flip, (input_direction[0] > 0, output_direction[0] > 0,
        #             input_direction[4] > 0, output_direction[4] > 0,
        #             input_direction[8] > 0, output_direction[8] > 0))
        origin = ref_image.GetOrigin()
        dist = (np.asarray(input_origin) - origin) / np.asarray(input_spacing)
        close_interp = np.floor(dist * np.asarray(input_spacing) / np.asarray(output_spacing))
        output_origin = ref_image.GetOrigin() + close_interp * np.asarray(output_spacing)
    else:
        output_spacing = [iso_voxel_size, iso_voxel_size, iso_voxel_size]
        output_origin = input_origin
        output_direction = input_direction
    output_size = np.ceil(np.asarray(input_size) * np.asarray(input_spacing) / np.asarray(output_spacing)).astype(int)


    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetSize(output_size.tolist())
    resample_filter.SetOutputDirection(output_direction)
    if is_label:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_image = resample_filter.Execute(input_image)
    else:
        resample_filter.SetInterpolator(sitk.sitkLinear)
        resampled_image = resample_filter.Execute(input_image)

    return resampled_image


class NoduleDataset(Dataset):
    """lung nodule CT image and contour dataset."""

    def __init__(
            self, 
            root_dir,
            load=False,
        ):
        """
        :param root_dir (string): directory location including all the images
        :param load (boolean, optional): to load the generated data offline for faster reading and not load RAM
        """
        self.root_dir = osp.expanduser(root_dir)
        self.samples = 0
        self.crop_size = (64, 64, 64)  # width, height, slice
        self.list = []
        self.threshold = 0.00000000001
        self.full_volume = None
        self.save_name = self.root_dir + '/list.txt'

        subvol = '_vol_iso' + str(self.crop_size[0]) + 'x' + str(self.crop_size[1]) + 'x' + str(self.crop_size[2])
        self.sub_vol_path = self.root_dir + '/generated' + subvol + '/'

        if load:
            ## load pre-generated data
            try:
                self.list = load_list(self.save_name)
                return
            except:
                pass

        self.create_input_data()

    def create_volumes(self, pid):
        # LIDC-IDRI-0001_CT_1-all.nrrd  <- CT image
        # LIDC-IDRI-0001_CT_1-all-ard.nrrd  <- Area Distortion Map
        # LIDC-IDRI-0001_CT_1-all-label.nrrd  <- Nodule Segmentation
        # LIDC-IDRI-0001_CT_1-all-peaks-label.nrrd  <- Spiculation:1, Lobulation: 2, Attachment: 3

        ard_file = glob.glob(f"{self.root_dir}/{pid}/{pid}_CT_*-ard.nrrd")[0]
        ard_image = sitk.ReadImage(ard_file)

        ct_file = ard_file.replace("-ard","")
        ct_image = sitk.ReadImage(ct_file)
        ct_spacing = np.asarray(ct_image.GetSpacing())
        iso_voxel_size = ct_spacing.min()

        label_file = ard_file.replace("-ard","-label")
        peak_label_file = ard_file.replace("-ard","-peaks-label")
        label = sitk.ReadImage(label_file)
        peak_label = sitk.ReadImage(peak_label_file)

        filename_prefix = self.sub_vol_path + pid + f"_iso{iso_voxel_size:.2f}_"

        # resample to be isotropic voxel
        ct_image = image_resample(ct_image, iso_voxel_size=iso_voxel_size)
        ard_image = image_resample(ard_image, iso_voxel_size=iso_voxel_size)
        label = image_resample(label, iso_voxel_size=iso_voxel_size, is_label=True)
        peak_label = image_resample(peak_label, iso_voxel_size=iso_voxel_size, is_label=True)\
        
        # crop or pad 
        diff_np = np.asarray(ct_image.GetSize()) - np.asarray(self.crop_size)
        #print(pid, ct_image.GetSize(), diff_np)
        diff_lower = np.ceil(diff_np/2).astype(int)
        diff_upper = np.floor(diff_np/2).astype(int)
        pad_lower = (-diff_lower*(diff_np<0)).tolist()
        pad_upper = (-diff_upper*(diff_np<0)).tolist()
        pad_lower[2] += int(self.crop_size[2]/2)
        pad_upper[2] += int(self.crop_size[2]/2)
        
        ct_image = sitk.ConstantPad(ct_image, pad_lower, pad_upper, -2000)
        ard_image = sitk.ConstantPad(ard_image, pad_lower, pad_upper, 0)
        label = sitk.ConstantPad(label, pad_lower, pad_upper, 0)
        peak_label = sitk.ConstantPad(peak_label, pad_lower, pad_upper, 0)

        self.list += create_sub_volumes(ct_image, ard_image, label, peak_label,
                                        samples=self.samples, crop_size=self.crop_size,
                                        filename_prefix=filename_prefix, th_percent=self.threshold)

    def create_input_data(self):
        make_dirs(self.sub_vol_path)

        studies = glob.glob(f"{self.root_dir}/*/")
        studies = [study for study in studies if study.find("generated") < 0] # remove generated folders
        #print(studies)
        print(osp.basename(self.root_dir), "Dataset: ", len(studies))
        with tqdm(studies) as pbar:
            for study in pbar:
                pid = osp.basename(study[:-1]) # '[:-1]' is to remove the last '/'
                pbar.set_description("Processing %s" % pid)
                self.create_volumes(pid)

        save_list(self.save_name, self.list)     

    def __len__(self):
        return len(self.list) 

    def __getitem__(self, index):
        pid, f_ct, f_ard, f_label, f_plabel = self.list[index]
        try:
            #ct_np, ard_np, label_np, plabel_np = np.load(f_ct), np.load(f_ard), np.load(f_label), np.load(f_plabel)
            ct_np, label_np = np.load(f_ct), np.load(f_label)
        except:
            print(pid)
            self.create_volumes(pid)
            #ct_np, ard_np, label_np, plabel_np = np.load(f_ct), np.load(f_ard), np.load(f_label), np.load(f_plabel)
            ct_np, label_np = np.load(f_ct), np.load(f_label)

        #return ct_np, ard_np, label_np, plabel_np
        return ct_np, label_np[0]


if __name__ == "__main__":
    data = NoduleDataset("DATA/LUNGx_spiculation")
