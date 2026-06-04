#!/usr/bin/env python3

"""
Generate nnU-Net segmentation figures for BraTS-MEN.

Default output:
    Case label | MRI | Ground truth | nnU-Net prediction | Error map

Default region:
    WT = whole tumour = labels {1, 2, 3}

BraTS-MEN labels:
    0 = background
    1 = NETC
    2 = SNFH
    3 = ET

BraTS evaluation regions:
    WT = {1, 2, 3}
    TC = {1, 3}
    ET = {3}

Default modality:
    channel 0001 = t1c / T1ce

Example:

python src/visualisation/make_report_figures.py \
    --repo /vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl \
    --metrics-csv /vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl/results/nnunet_brats_men_results/brats_men_metrics.csv \
    --channel 0001 \
    --region WT \
    --plane axial
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def strip_nii_gz(filename: str) -> str:
    """Convert BraTS-MEN-01426-000.nii.gz -> BraTS-MEN-01426-000."""
    filename = str(filename)

    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]

    return filename


def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI image as a NumPy array."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return nib.load(str(path)).get_fdata()


def robust_normalise(image_slice: np.ndarray) -> np.ndarray:
    """
    Normalise an MRI slice for display using percentiles.
    This avoids a few bright voxels dominating the display contrast.
    """
    image_slice = np.nan_to_num(image_slice)

    nonzero = image_slice[image_slice != 0]

    if nonzero.size > 0:
        lo, hi = np.percentile(nonzero, [1, 99])
    else:
        lo, hi = np.percentile(image_slice, [1, 99])

    if hi <= lo:
        return np.zeros_like(image_slice, dtype=np.float32)

    image_slice = np.clip(image_slice, lo, hi)
    image_slice = (image_slice - lo) / (hi - lo)

    return image_slice.astype(np.float32)


def get_slice(volume: np.ndarray, slice_idx: int, plane: str) -> np.ndarray:
    """Extract a 2D slice from a 3D volume."""
    if plane == "axial":
        sl = volume[:, :, slice_idx]
    elif plane == "coronal":
        sl = volume[:, slice_idx, :]
    elif plane == "sagittal":
        sl = volume[slice_idx, :, :]
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return np.rot90(sl)


def make_region_mask(seg: np.ndarray, region: str) -> np.ndarray:
    """
    Convert multiclass BraTS-MEN segmentation to a binary region mask.

    WT = whole tumour = labels 1, 2, 3
    TC = tumour core   = labels 1, 3
    ET = enhancing tumour = label 3
    """
    region = region.upper()

    if region == "WT":
        return seg > 0

    if region == "TC":
        return (seg == 1) | (seg == 3)

    if region == "ET":
        return seg == 3

    raise ValueError(
        f"Unknown region '{region}'. Use one of: WT, TC, ET."
    )


def choose_best_slice(
    gt_region: np.ndarray,
    pred_region: np.ndarray,
    plane: str,
    slice_source: str,
) -> int:
    """
    Choose the slice to visualise.

    slice_source:
        gt    = largest ground-truth region slice
        pred  = largest predicted region slice
        union = largest union of GT and prediction
    """
    if slice_source == "gt":
        mask = gt_region
    elif slice_source == "pred":
        mask = pred_region
    elif slice_source == "union":
        mask = gt_region | pred_region
    else:
        raise ValueError(f"Unknown slice_source: {slice_source}")

    if plane == "axial":
        areas = mask.sum(axis=(0, 1))
        fallback_idx = mask.shape[2] // 2
    elif plane == "coronal":
        areas = mask.sum(axis=(0, 2))
        fallback_idx = mask.shape[1] // 2
    elif plane == "sagittal":
        areas = mask.sum(axis=(1, 2))
        fallback_idx = mask.shape[0] // 2
    else:
        raise ValueError(f"Unknown plane: {plane}")

    if areas.max() == 0:
        return fallback_idx

    return int(np.argmax(areas))


def crop_around_mask(
    image_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    margin: int = 50,
):
    """
    Crop around the union of GT and prediction masks.

    This removes excessive black background and makes the tumour easier to see.
    """
    mask = gt_slice | pred_slice

    if mask.sum() == 0:
        mask = image_slice > 0.05

    coords = np.argwhere(mask)

    if coords.size == 0:
        return image_slice, gt_slice, pred_slice

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(int(y_min) - margin, 0)
    x_min = max(int(x_min) - margin, 0)
    y_max = min(int(y_max) + margin + 1, image_slice.shape[0])
    x_max = min(int(x_max) + margin + 1, image_slice.shape[1])

    return (
        image_slice[y_min:y_max, x_min:x_max],
        gt_slice[y_min:y_max, x_min:x_max],
        pred_slice[y_min:y_max, x_min:x_max],
    )


def maybe_draw_contour(
    ax,
    mask_slice: np.ndarray,
    color: str,
    linewidth: float = 1.8,
):
    """Draw a contour only if the mask has foreground voxels."""
    binary = mask_slice.astype(bool)

    if binary.sum() == 0:
        return

    ax.contour(
        binary.astype(np.float32),
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
    )


def make_error_overlay(gt_slice: np.ndarray, pred_slice: np.ndarray) -> np.ndarray:
    """
    Create RGBA error overlay.

    green = true positive
    red   = false positive
    blue  = false negative
    """
    gt = gt_slice.astype(bool)
    pred = pred_slice.astype(bool)

    tp = gt & pred
    fp = ~gt & pred
    fn = gt & ~pred

    overlay = np.zeros((*gt.shape, 4), dtype=np.float32)

    overlay[tp] = [0.0, 0.85, 0.0, 0.45]
    overlay[fp] = [1.0, 0.0, 0.0, 0.55]
    overlay[fn] = [0.0, 0.2, 1.0, 0.55]

    return overlay


def load_channel_names(dataset_json_path: Path) -> dict:
    """Read channel names from nnU-Net dataset.json if available."""
    if not dataset_json_path.exists():
        return {}

    with open(dataset_json_path, "r") as f:
        dataset = json.load(f)

    return dataset.get("channel_names", {})


def pretty_channel_name(channel_name: str) -> str:
    """Convert dataset channel names into report-friendly names."""
    mapping = {
        "t1n": "T1-native",
        "t1c": "T1ce",
        "t2w": "T2-weighted",
        "t2f": "T2-FLAIR",
    }

    return mapping.get(channel_name.lower(), channel_name)


def find_metrics_csv(repo: Path) -> Path | None:
    """Try common locations for brats_men_metrics.csv."""
    candidates = [
        repo / "brats_men_metrics.csv",
        repo / "reports" / "brats_men_metrics.csv",
        repo / "reports" / "tables" / "brats_men_metrics.csv",
        repo / "data" / "brats_men_metrics.csv",
        repo / "data" / "evaluation" / "brats_men_metrics.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_case_metric_lookup(
    metrics_csv: Path | None,
    metric_label: str,
    metric_column: str,
) -> dict:
    """
    Load metric values from CSV.

    Expected columns:
        Case
        Labels
        Legacy_Dice

    Returns:
        {
            "BraTS-MEN-01426-000": 0.995,
            ...
        }
    """
    if metrics_csv is None:
        return {}

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    df = pd.read_csv(metrics_csv)

    required_columns = {"Case", "Labels", metric_column}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Metrics CSV is missing required columns: {sorted(missing)}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df.copy()
    df["Case_clean"] = df["Case"].astype(str).apply(strip_nii_gz)
    df["Labels"] = df["Labels"].astype(str)

    label_df = df[df["Labels"] == metric_label].copy()

    if label_df.empty:
        available_labels = sorted(df["Labels"].unique().tolist())
        raise ValueError(
            f"No rows found for label '{metric_label}'. "
            f"Available labels are: {available_labels}"
        )

    metric_lookup = {}

    for _, row in label_df.iterrows():
        case_id = row["Case_clean"]
        value = row[metric_column]

        if pd.isna(value):
            continue

        metric_lookup[case_id] = float(value)

    return metric_lookup


def format_case_label(
    case_id: str,
    metric_lookup: dict,
    region: str,
    metric_name_for_display: str,
) -> str:
    """
    Create row label for the report figure.

    Example:
        BraTS-MEN-01426-000
        WT Dice = 0.995
    """
    lines = [case_id]

    if case_id in metric_lookup:
        lines.append(
            f"{region} {metric_name_for_display} = {metric_lookup[case_id]:.3f}"
        )

    return "\n".join(lines)


def prepare_case(
    case_id: str,
    display_label: str,
    image_path: Path,
    label_path: Path,
    pred_path: Path,
    plane: str,
    region: str,
    slice_source: str,
    crop_margin: int,
):
    """Load one case and return display-ready slices."""
    image = load_nifti(image_path)
    gt_seg = load_nifti(label_path)
    pred_seg = load_nifti(pred_path)

    gt_region = make_region_mask(gt_seg, region)
    pred_region = make_region_mask(pred_seg, region)

    slice_idx = choose_best_slice(
        gt_region=gt_region,
        pred_region=pred_region,
        plane=plane,
        slice_source=slice_source,
    )

    image_slice = get_slice(image, slice_idx, plane)
    gt_slice = get_slice(gt_region, slice_idx, plane)
    pred_slice = get_slice(pred_region, slice_idx, plane)

    image_slice = robust_normalise(image_slice)

    image_slice, gt_slice, pred_slice = crop_around_mask(
        image_slice=image_slice,
        gt_slice=gt_slice,
        pred_slice=pred_slice,
        margin=crop_margin,
    )

    return {
        "case_id": case_id,
        "display_label": display_label,
        "slice_idx": slice_idx,
        "image_slice": image_slice,
        "gt_slice": gt_slice,
        "pred_slice": pred_slice,
    }


def make_panel(
    prepared_cases: list,
    output_path: Path,
    plane: str,
    channel: str,
    channel_name: str,
    region: str,
    metric_name_for_display: str,
):
    """Create the final report-ready multi-row panel."""
    n_cases = len(prepared_cases)

    fig, axes = plt.subplots(
        n_cases,
        5,
        figsize=(16, 3.6 * n_cases),
        squeeze=False,
        gridspec_kw={
            "width_ratios": [1.25, 2.2, 2.2, 2.2, 2.2],
            "wspace": 0.04,
            "hspace": 0.05,
        },
    )

    column_titles = [
        "",
        "MRI",
        "Ground truth",
        "nnU-Net prediction",
        "Error map",
    ]

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=13, pad=8)

    for row_idx, case in enumerate(prepared_cases):
        image_slice = case["image_slice"]
        gt_slice = case["gt_slice"]
        pred_slice = case["pred_slice"]

        row_label = (
            case["display_label"]
            + f"\n{plane} slice {case['slice_idx']}"
        )

        # Case label column
        ax = axes[row_idx, 0]
        ax.text(
            0.98,
            0.5,
            row_label,
            fontsize=13,
            ha="right",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")

        # MRI only
        ax = axes[row_idx, 1]
        ax.imshow(image_slice, cmap="gray")
        ax.axis("off")

        # Ground-truth contour
        ax = axes[row_idx, 2]
        ax.imshow(image_slice, cmap="gray")
        maybe_draw_contour(ax, gt_slice, color="red")
        ax.axis("off")

        # Prediction contour
        ax = axes[row_idx, 3]
        ax.imshow(image_slice, cmap="gray")
        maybe_draw_contour(ax, pred_slice, color="cyan")
        ax.axis("off")

        # Error map
        ax = axes[row_idx, 4]
        ax.imshow(image_slice, cmap="gray")
        ax.imshow(make_error_overlay(gt_slice, pred_slice), interpolation="none")
        ax.axis("off")

    legend_handles = [
        Patch(facecolor="limegreen", edgecolor="none", alpha=0.6, label="True positive"),
        Patch(facecolor="red", edgecolor="none", alpha=0.6, label="False positive"),
        Patch(facecolor="blue", edgecolor="none", alpha=0.6, label="False negative"),
        Patch(facecolor="none", edgecolor="red", label="GT contour"),
        Patch(facecolor="none", edgecolor="cyan", label="Prediction contour"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.005),
    )

    pretty_modality = pretty_channel_name(channel_name) if channel_name else f"channel {channel}"

    fig.suptitle(
        f"Qualitative nnU-Net segmentation examples on BraTS-MEN using {pretty_modality} MRI",
        fontsize=14,
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0.055, 1, 0.965])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate report-ready nnU-Net qualitative segmentation figures."
    )

    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/vol/biomedic2/bglocker_studproj/am525/3d-brain-ssl"),
        help="Path to repository root.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset001_BraTSMEN",
        help="nnU-Net dataset name.",
    )

    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "BraTS-MEN-01426-000",
            "BraTS-MEN-01166-000",
            "BraTS-MEN-00362-000",
        ],
        help="Case IDs to visualise.",
    )

    parser.add_argument(
        "--channel",
        type=str,
        default="0001",
        help="nnU-Net image channel to display. For your dataset, 0001 = t1c/T1ce.",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="WT",
        choices=["WT", "TC", "ET"],
        help="Segmentation region to visualise. Default WT = labels 1,2,3.",
    )

    parser.add_argument(
        "--plane",
        type=str,
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="Anatomical plane to visualise.",
    )

    parser.add_argument(
        "--slice-source",
        type=str,
        default="gt",
        choices=["gt", "pred", "union"],
        help="How to choose the slice. Default uses largest GT region slice.",
    )

    parser.add_argument(
        "--crop-margin",
        type=int,
        default=50,
        help="Pixel margin around tumour crop.",
    )

    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Path to brats_men_metrics.csv. If omitted, common repo locations are searched.",
    )

    parser.add_argument(
        "--metric-column",
        type=str,
        default="Legacy_Dice",
        help="Metric column to display from CSV.",
    )

    parser.add_argument(
        "--metric-name-for-display",
        type=str,
        default="Dice",
        help="Metric name shown on figure labels.",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing nnU-Net imagesTs files.",
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory containing labelsTs files.",
    )

    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Directory containing nnU-Net predictions.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for report figures.",
    )

    args = parser.parse_args()

    images_dir = (
        args.images_dir
        or args.repo / "data" / "nnUNet_raw" / args.dataset / "imagesTs"
    )

    labels_dir = (
        args.labels_dir
        or args.repo / "data" / "nnUNet_raw" / args.dataset / "labelsTs"
    )

    pred_dir = (
        args.pred_dir
        or args.repo / "data" / "nnunet_predictions_postprocessed"
    )

    out_dir = (
        args.out_dir
        or args.repo / "reports" / "figures" / "nnunet_examples"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_json_path = args.repo / "data" / "nnUNet_raw" / args.dataset / "dataset.json"
    channel_names = load_channel_names(dataset_json_path)

    channel_key = str(int(args.channel))
    channel_name = channel_names.get(channel_key, "")

    metrics_csv = args.metrics_csv
    if metrics_csv is None:
        metrics_csv = find_metrics_csv(args.repo)

    metric_lookup = load_case_metric_lookup(
        metrics_csv=metrics_csv,
        metric_label=args.region,
        metric_column=args.metric_column,
    )

    print("=== nnU-Net report figure generation ===")
    print(f"Repo:          {args.repo}")
    print(f"Dataset:       {args.dataset}")
    print(f"Images dir:    {images_dir}")
    print(f"Labels dir:    {labels_dir}")
    print(f"Pred dir:      {pred_dir}")
    print(f"Output dir:    {out_dir}")
    print(f"Channel:       {args.channel} {f'({channel_name})' if channel_name else ''}")
    print(f"Region:        {args.region}")
    print(f"Plane:         {args.plane}")
    print(f"Slice source:  {args.slice_source}")
    print(f"Crop margin:   {args.crop_margin}")
    print(f"Metrics CSV:   {metrics_csv if metrics_csv else 'not found / not used'}")
    print(f"Metric shown:  {args.region} {args.metric_name_for_display}")
    print()

    prepared_cases = []

    for raw_case_id in args.cases:
        case_id = strip_nii_gz(raw_case_id)

        image_path = images_dir / f"{case_id}_{args.channel}.nii.gz"
        label_path = labels_dir / f"{case_id}.nii.gz"
        pred_path = pred_dir / f"{case_id}.nii.gz"

        display_label = format_case_label(
            case_id=case_id,
            metric_lookup=metric_lookup,
            region=args.region,
            metric_name_for_display=args.metric_name_for_display,
        )

        print(f"Processing {case_id}")
        print(f"  image: {image_path}")
        print(f"  label: {label_path}")
        print(f"  pred:  {pred_path}")

        if case_id in metric_lookup:
            print(
                f"  {args.region} {args.metric_name_for_display}: "
                f"{metric_lookup[case_id]:.3f}"
            )
        else:
            print("  metric: not found in CSV")

        case_data = prepare_case(
            case_id=case_id,
            display_label=display_label,
            image_path=image_path,
            label_path=label_path,
            pred_path=pred_path,
            plane=args.plane,
            region=args.region,
            slice_source=args.slice_source,
            crop_margin=args.crop_margin,
        )

        print(f"  selected slice: {case_data['slice_idx']}")
        prepared_cases.append(case_data)

    pretty_modality = pretty_channel_name(channel_name) if channel_name else f"channel{args.channel}"

    output_name = (
        f"nnunet_report_panel_"
        f"{args.plane}_"
        f"{pretty_modality.replace('-', '').replace(' ', '')}_"
        f"{args.region}_{args.metric_name_for_display}"
    )

    output_path = out_dir / f"{output_name}.png"

    make_panel(
        prepared_cases=prepared_cases,
        output_path=output_path,
        plane=args.plane,
        channel=args.channel,
        channel_name=channel_name,
        region=args.region,
        metric_name_for_display=args.metric_name_for_display,
    )

    print()
    print("Saved final report figure:")
    print(output_path)


if __name__ == "__main__":
    main()