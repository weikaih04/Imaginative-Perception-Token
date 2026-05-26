# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
}


# ============================================================================
# Spatial Reasoning training data (MVC + PET).
#
# Each entry below corresponds to one HuggingFace dataset hosted at
# huggingface.co/datasets/weikaih/<name>. Download them once into
# `data/training/<name>/` with `python scripts/download_spatial_datasets.py`
# (a thin wrapper around `huggingface_hub.snapshot_download`).
#
# Collection: https://huggingface.co/collections/weikaih/spatial-imaginative-token-mvc-pet-datasets
# ============================================================================

import os

# Local download root. Override via DATA_ROOT env var if you want to keep
# parquets elsewhere.
_DATA_ROOT = os.environ.get(
    'DATA_ROOT',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'training'),
)


def _spatial_entry(name, num_files, num_total_samples):
    """Helper for the MVC + PET datasets; all share the same folder layout."""
    data_dir = os.path.join(_DATA_ROOT, name)
    return {
        'data_dir': data_dir,
        'num_files': num_files,
        'num_total_samples': num_total_samples,
        'parquet_info_path': os.path.join(data_dir, 'parquet_info.json'),
    }


DATASET_INFO = {
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},

        # ====== Multi-View Counting (MVC) ======
        # HF: weikaih/spatial-imaginative-token-mvc-ipt
        'multi_view_counting':       _spatial_entry('multi_view_counting',       5, 17079),
        # HF: weikaih/spatial-imaginative-token-mvc-no-thought
        'mvc_no_thought':            _spatial_entry('mvc_no_thought',            5, 17079),
        # HF: weikaih/spatial-imaginative-token-mvc-textcot
        'mvc_textcot':               _spatial_entry('mvc_textcot',               5, 16808),
        # HF: weikaih/spatial-imaginative-token-mvc-answeronly
        'mvc_answeronly':            _spatial_entry('mvc_answeronly',            5, 17079),

        # ====== Perspective Taking (PET) ======
        # HF: weikaih/spatial-imaginative-token-pet-ipt
        'perspective_with_sysprompt': _spatial_entry('perspective_with_sysprompt', 5, 20531),
        # HF: weikaih/spatial-imaginative-token-pet-no-thought
        'perspective_no_thought':    _spatial_entry('perspective_no_thought',    5, 20531),
        # HF: weikaih/spatial-imaginative-token-pet-textcot
        'perspective_textcot':       _spatial_entry('perspective_textcot',       5, 20531),
        # HF: weikaih/spatial-imaginative-token-pet-answeronly
        'pet_answeronly':            _spatial_entry('pet_answeronly',            5, 20531),
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
}