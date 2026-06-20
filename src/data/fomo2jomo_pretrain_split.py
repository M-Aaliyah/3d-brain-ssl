import os
import logging
from utils.fomo2jomo_utils import load_data_pickle, save_data_pickle, parse_dataset_structure, parse_dataset_preprocessed
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, isfile, save_pickle, load_pickle
from yucca.pipeline.configuration.configure_paths import PathConfig
from yucca.pipeline.configuration.split_data import SplitConfig, simple_split, split_is_precomputed, get_file_names


def get_pretrain_split_config(method: str, idx: int, split_ratio: float, path_config: PathConfig, per_subject=False, path_dict_structure=None):
    splits_path = join(path_config.task_dir, "fomo2jomo_splits.pkl")

    assert method in [
        "simple_train_val_split",
        "multi_sequence_simple_train_val_split",
    ], "this module only supports a subset of the split methods"

    if isfile(splits_path):
        splits = load_pickle(splits_path)
        assert isinstance(splits, dict)

        if split_is_precomputed(splits, method, idx):
            logging.warning(
                f"Reusing already computed split file which was split using the {method} method and parameter {split_ratio}."
            )
            return SplitConfig(splits, method, idx)
        else:
            logging.warning("Generating new split since splits did not contain a split computed with the same parameters.")
    else:
        splits = {}

    if method not in splits.keys():
        splits[method] = {}

    assert method == "simple_train_val_split"

    if per_subject:
        if path_dict_structure is None or not isfile(path_dict_structure):            
            parent_dir = os.path.dirname(path_config.train_data_dir)
            path_dict_structure = os.path.join(parent_dir, 'subject_level_dict.pkl')
            data_dict = parse_dataset_preprocessed(path_config.train_data_dir)
            save_data_pickle(data_dict, path_dict_structure)
        else:
            data_dict = load_data_pickle(path_dict_structure)
        subject_names = list(data_dict.keys())
        # [{"train": list(file_names[num_val:]), "val": list(file_names[:num_val])}] - return object format
        splits[method][split_ratio] = simple_split(subject_names, split_ratio)
    else:
        # Every file is treated as a separate subject, so we can just get the file names and split them
        names = get_file_names(path_config.train_data_dir)
        splits[method][split_ratio] = simple_split(names, split_ratio)  # type: ignore
        data_dict = None
        
    split_cfg = SplitConfig(splits, method, split_ratio)
    save_pickle(splits, splits_path)

    return split_cfg, data_dict