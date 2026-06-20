"""From the fomo2jomo repository."""
from curses import meta
import torchvision
import numpy as np
import random
import torch
import os
import copy
from torch.utils.data import Dataset
from typing import Tuple, Optional, Literal
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad
from yucca.modules.data.augmentation.transforms.formatting import NumpyToTorch

from batchgenerators.utilities.file_and_folder_operations import join

# Custom cropping to allow to pass an index
from utils.padding import croppad_3D_case_from_3D


class FOMODataset(Dataset):
    """
    Dataset class for FOMO downstream tasks. Supports classification and regression tasks.
    For segmentation tasks, use YuccaTrainDataset from the Yucca library instead.
    """

    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
        task_type: Literal["classification", "regression"] = "classification",
        per_subject = False,
        dataset_dict = None,
        allow_missing_modalities: Optional[bool] = False,  # For compatibility
        p_oversample_foreground: Optional[float] = None,  # For compatibility
        crop=False,
    ):
        super().__init__()
        # Support only non-segmentation tasks
        assert task_type in [
            "classification",
            "regression",
        ], f"Unsupported task type: {task_type}. For segmentation use YuccaTrainDataset instead."

        self.task_type = task_type
        self.all_files = samples
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.per_subject = per_subject
        self.dataset_dict = dataset_dict
        self.crop = crop
        self.croppad = CropPad(patch_size=self.patch_size)
        self.to_torch = NumpyToTorch()

    def load_scan(self, subject_id, session_id, scan_filename):
        scan_name = scan_filename.replace('.nii.gz', '')
        case = f"{subject_id}_{session_id}_{scan_name}"
        data = self._load_volume(case)
        if np.isnan(data).any() or np.isinf(data).any():
            if "DISABLE_NAN_WARNING" not in os.environ:
                print(f"A case contains NaNs or infs: {case}")
            raise FileNotFoundError
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
        return data, case

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx, retries=0):
        if retries > 5:
            raise RuntimeError("Too many failed attempts to load data.")
        
        try:
            case = self.all_files[idx]

            # single modality
            assert isinstance(case, str)

            data = self._load_volume(case)
            label = self._load_label(case)
            if self.task_type == "classification":
                try:
                    seg = self._load_seg(case)
                except:                    
                    # To not change the label
                    seg = np.ones_like(data)
            else:
                seg = np.ones_like(data)

            # print(data.shape)
            data_dict = {
                "file_path": case,
                "image": data,
                "seg": seg,
                "label": label,
            }

            metadata = {"foreground_locations": []}
            return self._transform(data_dict, metadata)
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self), retries=retries+1)

    def _transform(self, data_dict, metadata=None):        
        label = data_dict["label"]
        seg = data_dict["seg"]

        if self.crop:
            input_shape, target_image_shape, target_label_shape, pad_kwargs = self.croppad.get_params(data_dict['image'], self.croppad.pad_value, self.patch_size)
            p_oversample_foreground = 0.0
            crop_idx = None
            image, _, crop_start_idx = croppad_3D_case_from_3D(data_dict["image"], metadata, None, 
                                                                 self.patch_size, p_oversample_foreground, target_image_shape=target_image_shape,
                                                                 target_label_shape=target_label_shape, crop_start_idx=crop_idx, **pad_kwargs)
            data_dict["image"] = image
            
            # Now, apply same crop to the seg.
            seg, _, crop_start_idx = croppad_3D_case_from_3D(data_dict["seg"], metadata, None, 
                                                             self.patch_size, p_oversample_foreground, target_image_shape=target_image_shape,
                                                             target_label_shape=target_label_shape, crop_start_idx=crop_start_idx, **pad_kwargs)
            data_dict["seg"] = seg

        # data_dict2["image"] = seg        
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        
        # NOTE: Remember that in the case that we don't have a mask, we provide an array full of ones, so the label does not change.
        # The logic is that if the label is positive but there is no lesion in the patch -> label = 0. We don't consider the opposite case.
        # data_dict["seg"] = data_dict["seg"].sum(axis=(1,2,3)) # We just want to know if there is a lesion in the patch or not.
        data_dict["seg"] = data_dict["seg"].sum() # It is per-subject the label, since now we evaluate the full case.
        data_dict["label"] = label

        return self.to_torch(data_dict)

    def _load_volume_and_header(self, file):
        vol = self._load_volume(file)
        header = load_pickle(file[: -len(".npy")] + ".pkl")
        return vol, header

    def _load_label(self, file):
        # For classification and regression, labels are in .txt files
        txt_file = file + ".txt"
        if self.task_type == "classification":
            return np.loadtxt(txt_file, dtype=int)
        else:  # regression
            reg_label = np.loadtxt(txt_file, dtype=float)
            reg_label = np.atleast_1d(reg_label)
            return reg_label

    def _load_volume(self, file):
        file = file + ".npy"

        try:
            vol = np.load(file, "r")
        except ValueError:
            vol = np.load(file, allow_pickle=True)

        return vol

    def _load_seg(self, file):
        file = file + "_seg.npy"

        try:
            vol = np.load(file, "r")
        except ValueError:
            vol = np.load(file, allow_pickle=True)

        return vol


class PretrainDataset(Dataset):
    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        data_dir: str,
        per_subject = False,
        dataset_dict = None,
        pre_aug_patch_size: Optional[Tuple[int, int, int]] = None,
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        self.all_files = samples
        self.data_dir = data_dir
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.pre_aug_patch_size = pre_aug_patch_size
        self.per_subject = per_subject
        self.dataset_dict = dataset_dict
        
        self.croppad = CropPad(patch_size=self.pre_aug_patch_size or self.patch_size)
        self.to_torch = NumpyToTorch()
    
    def load_scan(self, subject_id, session_id, scan_filename):
        # scan_name = scan_filename.replace('.nii.gz', '')
        # case = f"{subject_id}_{session_id}_{scan_name}"
        data = self._load_volume(scan_filename)
        if np.isnan(data).any() or np.isinf(data).any():
            if "DISABLE_NAN_WARNING" not in os.environ:
                print(f"A case contains NaNs or infs: {scan_filename}")
            raise FileNotFoundError
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
        return data, scan_filename

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx, retries=0):
        if retries > 5:
            raise RuntimeError("Too many failed attempts to load data.")
        
        try:
            if self.per_subject:
                # Get subject ID
                subject_id = self.all_files[idx]
        
                # Randomly choose a session
                sessions = list(self.dataset_dict[subject_id].keys())
                session_id = random.choice(sessions)
        
                # Randomly choose a couple of scans in that session
                scans = self.dataset_dict[subject_id][session_id]

                # Handle case with only one scan (return x_b=None)
                scan_a = random.choice(scans)
                scan_b = random.choice([s for s in scans if s != scan_a]) if len(scans) > 1 else None
            
                x_a, case_a = self.load_scan(subject_id, session_id, scan_a)
                x_b, case_b = (None, None)
                if scan_b is not None:
                    x_b, case_b = self.load_scan(subject_id, session_id, scan_b)
                else:
                    # Simulate a 2nd image of the same channel/modality. To simplify training logic.
                    x_b = copy.copy(x_a)
                    case_b = case_a

                metadata = {"foreground_locations": []}
                data_dict_a = {"file_path": case_a}
                data_dict_a["image"] = x_a

                data_dict_b = {"file_path": case_b}
                data_dict_b["image"] = x_b
                
                # NOTE: A few times it is not exactly the same! But most of the time they are co-registered.
                # Important assumption: ASSUME IMAGES ARE MOSTLY CO-REGISTERED
                # print(f"Size A: {x_a.shape}, Size B: {x_b.shape}"
                data_dict_a, crop_idx = self._transform(data_dict_a, metadata, crop_idx=None)                
                data_dict_b, crop_idx = self._transform(data_dict_b, metadata, crop_idx=crop_idx)
                return {
                    "x_a": data_dict_a,
                    "x_b": data_dict_b,
                }
            else:
                # Now, leave this here just for compatbility
                case = self.all_files[idx]

                # single modality
                assert isinstance(case, str)
                data = self._load_volume(case)

                # Ensure volume does not contain NaNs or Infs, which can sometimes
                # occur in large pretraining datasets.
                if np.isnan(data).any() or np.isinf(data).any():
                    if "DISABLE_NAN_WARNING" not in os.environ:
                        print("A case contains NaNs or infs. We have corrected this, but consider handling this with different preprocessing or skipping affected cases.")
                        print(f"Affected Case: {case}")
                        print("Set DISABLE_NAN_WARNING=1 to disable this warning.")
                    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
                
                metadata = {"foreground_locations": []}
                data_dict = {"file_path": case}
                data_dict["image"] = data

                data_dict_b = {"file_path": None}
                data_dict_b["image"] = None

                return {
                    "x_a": self._transform(data_dict, metadata),
                    "x_b": self._transform(data_dict_b, metadata),
                }

        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self), retries=retries+1)

    def _transform(self, data_dict, metadata=None, crop_idx=None):
        if data_dict['image'] is not None:
            # NOTE: We change the croppad to ours, in order to be able to provide the croppidx, we want the same patch in all sequences/modalities.
            # data_dict = self.croppad(data_dict, metadata)
            patch_size = self.pre_aug_patch_size or self.patch_size
            p_oversample_foreground = 0.0
            input_shape, target_image_shape, target_label_shape, pad_kwargs = self.croppad.get_params(data_dict['image'], self.croppad.pad_value, patch_size)
            target_image_shape, target_label_shape
            if data_dict.get("label") is not None:
                label = data_dict["label"]
            else:
                label = None            
            image, label, crop_start_idx = croppad_3D_case_from_3D(data_dict["image"], metadata, label, 
                                                                   patch_size, p_oversample_foreground, target_image_shape=target_image_shape,
                                                                   target_label_shape=target_label_shape, crop_start_idx=crop_idx, **pad_kwargs)
            data_dict["image"] = image
            data_dict["label"] = label
            if self.composed_transforms is not None:
                data_dict = self.composed_transforms(data_dict)
            return self.to_torch(data_dict), crop_start_idx
        else:
            return self.to_torch(data_dict), None

    def _load_volume_and_header(self, file):
        vol = self._load_volume(file)
        header = load_pickle(file[: -len(".npy")] + ".pkl")
        return vol, header

    def _load_volume(self, file):
        if not file.endswith('.npy'):
            file = file + ".npy"
        path = join(self.data_dir, file)

        try:
            vol = np.load(path, "r")
        except ValueError:
            vol = np.load(path, allow_pickle=True)

        # Add channel dimension if it doesn't exist
        if len(vol.shape) == 3:
            vol = vol[np.newaxis, ...]

        return vol