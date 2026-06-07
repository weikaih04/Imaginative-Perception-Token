#!/usr/bin/env python3
"""Download all MVC + PET training datasets from HuggingFace into ./data/training/.

Usage:
    python release/scripts/download_datasets.py
    python release/scripts/download_datasets.py --task mvc           # MVC only
    python release/scripts/download_datasets.py --task pet           # PET only
    python release/scripts/download_datasets.py --names mvc_answeronly perspective_no_thought
"""

import argparse
import os
from huggingface_hub import snapshot_download

# (local_name, hf_repo_id, task)
DATASETS = [
    # MVC
    ("multi_view_counting",       "weikaih/imaginative-perception-token-mvc-ipt",          "mvc"),
    ("mvc_no_thought",            "weikaih/imaginative-perception-token-mvc-no-thought",                           "mvc"),
    ("mvc_textcot",               "weikaih/imaginative-perception-token-mvc-textcot",                              "mvc"),
    ("mvc_answeronly",            "weikaih/imaginative-perception-token-mvc-answeronly",                           "mvc"),
    # PET
    ("perspective_with_sysprompt", "weikaih/imaginative-perception-token-pet-ipt", "pet"),
    ("perspective_no_thought",    "weikaih/imaginative-perception-token-pet-no-thought",                   "pet"),
    ("perspective_textcot",       "weikaih/imaginative-perception-token-pet-textcot",                      "pet"),
    ("pet_answeronly",            "weikaih/imaginative-perception-token-pet-answeronly",                           "pet"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mvc", "pet", "all"], default="all")
    parser.add_argument("--names", nargs="*", default=None,
                        help="Specific local dataset names to download.")
    parser.add_argument("--data-root", default="data/training",
                        help="Local root directory for datasets (default: data/training)")
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)

    for local_name, hf_repo, task in DATASETS:
        if args.task != "all" and task != args.task:
            continue
        if args.names and local_name not in args.names:
            continue
        local_dir = os.path.join(args.data_root, local_name)
        print(f"[download] {hf_repo} -> {local_dir}")
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    print("Done.")


if __name__ == "__main__":
    main()
