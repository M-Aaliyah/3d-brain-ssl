"""From https://github.com/jbanusco/fomo25"""
import lightning as pl
import torchvision
import logging
import torch
from typing import Literal, Optional, Union
from torch.utils.data import DataLoader, Sampler
from batchgenerators.utilities.file_and_folder_operations import join
from yucca.pipeline.configuration.split_data import SplitConfig
from yucca.modules.data.datasets.YuccaDataset import YuccaTestDataset
from yucca.modules.data.samplers import InfiniteRandomSampler
from monai.data import pad_list_data_collate

from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from data.yucca_dataset import ModYuccaTrainDataset


class ModYuccaDataModule(YuccaDataModule):
    def __init__(
        self,
        batch_size: int,
        patch_size: tuple,
        allow_missing_modalities: Optional[bool] = False,
        image_extension: Optional[str] = None,
        composed_train_transforms: Optional[torchvision.transforms.Compose] = None,
        composed_val_transforms: Optional[torchvision.transforms.Compose] = None,
        num_workers: Optional[int] = None,
        overwrite_predictions: bool = False,
        pred_data_dir: Optional[str] = None,
        pred_include_cases: Optional[list] = None,
        pred_save_dir: Optional[str] = None,
        pre_aug_patch_size: Optional[Union[list, tuple]] = None,
        p_oversample_foreground: Optional[float] = 0.33,
        splits_config: Optional[SplitConfig] = None,
        split_idx: Optional[int] = None,
        task_type: Optional[str] = None,
        test_dataset_class: Optional[torch.utils.data.Dataset] = YuccaTestDataset,
        train_data_dir: Optional[str] = None,
        train_dataset_class: Optional[torch.utils.data.Dataset] = ModYuccaTrainDataset,
        train_sampler: Optional[Sampler] = InfiniteRandomSampler,
        val_sampler: Optional[Sampler] = InfiniteRandomSampler,
    ):
        super().__init__(batch_size, patch_size, allow_missing_modalities, image_extension, composed_train_transforms, composed_val_transforms, 
                         num_workers, overwrite_predictions, pred_data_dir, pred_include_cases, pred_save_dir, pre_aug_patch_size, 
                         p_oversample_foreground, splits_config, split_idx, task_type, test_dataset_class, train_data_dir, train_dataset_class,
                         train_sampler, val_sampler)
        

    def setup(self, stage: Literal["fit", "test", "predict"]):
        # Mod setup, to use our ModTrainer without cropping

        logging.info(f"Setting up data for stage: {stage}")
        expected_stages = ["fit", "test", "predict"]
        assert stage in expected_stages, "unexpected stage. " f"Expected: {expected_stages} and found: {stage}"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            assert self.train_data_dir is not None
            assert self.split_idx is not None
            assert self.splits_config is not None
            assert self.task_type is not None

            self.train_samples = [join(self.train_data_dir, i) for i in self.splits_config.train(self.split_idx)]
            self.val_samples = [join(self.train_data_dir, i) for i in self.splits_config.val(self.split_idx)]

            if len(self.train_samples) < 100:
                logging.info(f"Training on samples: {self.train_samples}")

            if len(self.val_samples) < 100:
                logging.info(f"Validating on samples: {self.val_samples}")

            self.train_dataset = self.train_dataset_class(
                self.train_samples,
                composed_transforms=self.composed_train_transforms,
                patch_size=self.pre_aug_patch_size if self.pre_aug_patch_size is not None else self.patch_size,
                task_type=self.task_type,
                allow_missing_modalities=self.allow_missing_modalities,
                p_oversample_foreground=self.p_oversample_foreground,
                crop=True,
            )  # At training we crop, it is ok.

            self.val_dataset = self.train_dataset_class(
                self.val_samples,
                composed_transforms=self.composed_val_transforms,
                patch_size=self.patch_size,
                task_type=self.task_type,
                allow_missing_modalities=self.allow_missing_modalities,
                p_oversample_foreground=self.p_oversample_foreground,
                crop=False,
            ) # At validation we don't crop! we will have the sliding window.

        if stage == "predict":
            assert self.pred_data_dir is not None, "`pred_data_dir` is required in inference"
            assert self.pred_save_dir is not None, "`pred_save_dir` is required in inference"
            assert self.image_extension is not None, "`image_extension` is required in inference"
            # This dataset contains ONLY the images (and not the labels)
            # It will return a tuple of (case, case_id)
            self.pred_dataset = self.test_dataset_class(
                self.pred_data_dir,
                pred_save_dir=self.pred_save_dir,
                overwrite_predictions=self.overwrite_predictions,
                suffix=self.image_extension,
                pred_include_cases=self.pred_include_cases,
            )

    def val_dataloader(self):
        # Use 1 in the case of segementation to compute the dice score properly
        # batch_size = 1 if self.task_type == "segmentation" else self.batch_size
        batch_size = 1  # We will always evaluate case per case using a sliding window over the patches
        collate_fn = pad_list_data_collate  # pads to the largest shape in the batch
        sampler = self.val_sampler(self.val_dataset) if self.val_sampler is not None else None
        return DataLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,            
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
        )