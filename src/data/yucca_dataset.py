"""From https://github.com/jbanusco/fomo25"""
import numpy as np
from typing import Union, Literal, Optional

from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset


class ModYuccaTrainDataset(YuccaTrainDataset):
    def __init__(
        self,
        samples: list,
        patch_size: list | tuple,
        keep_in_ram: Union[bool, None] = None,
        label_dtype: Optional[Union[int, float]] = None,
        composed_transforms=None,
        task_type: Literal["classification", "segmentation", "self-supervised", "contrastive"] = "segmentation",
        allow_missing_modalities=False,
        p_oversample_foreground=0.33,
        crop=False,
    ):
        super().__init__(samples, patch_size, keep_in_ram, label_dtype, composed_transforms, task_type, allow_missing_modalities, p_oversample_foreground)
        self.crop = crop
        
    def _transform(self, data_dict, metadata):
        if self.crop:            
            data_dict = self.croppad(data_dict, metadata)
        else:
            data_dict['image'] = data_dict['image'].astype(np.float32)
            data_dict['label'] = data_dict['label'].astype(np.float32)

        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        return self.to_torch(data_dict)