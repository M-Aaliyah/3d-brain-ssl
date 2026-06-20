"""
Configuration for BraTS-MEN dataset. This is a multi-class segmentation dataset for brain tumor segmentation.

BraTS-MEN details:
  - 4 MRI modalities: T1, T1ce (T1 post-contrast), T2, T2-FLAIR
  - 4 classes: 0=background, 1=NCR (necrotic core), 2=ED (edema), 3=ET (enhancing tumour)
  - Compound regions evaluated by brats_men_metrics.py:
      WT (whole tumour) = labels 1+2+3
      TC (tumour core)  = labels 1+3
      ET (enhancing)    = label 3
  - Spacing: already 1x1x1mm in BraTS data, so no resampling needed
    but we keep target_spacing=[1.0,1.0,1.0] to match fomo25 pipeline exactly
"""

bratsmen_config = {
    # Preprocessed data folder and for logging
    "task_name": "Task001_BraTSMEN",

    "crop_to_nonzero": True,
    "deep_supervision": False,
    "keep_aspect_ratio": True,
    "norm_op": "volume_wise_znorm",

    "modalities": ("T1", "T1ce", "T2", "T2FLAIR"),

    # Segmentation classes and labels
    "num_classes": 4,
    "task_type": "segmentation",
    "label_extension": ".nii.gz",

    # Label map for BraTS-MEN dataset
    "labels": {
        0: "background",
        1: "NCR",    # necrotic core
        2: "ED",     # peritumoral edema
        3: "ET",     # enhancing tumour
    },
}