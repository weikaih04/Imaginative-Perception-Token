#!/usr/bin/env python
"""Prepare downloaded HuggingFace datasets for training.

The released HF datasets are stored in a viewer-friendly raw schema (question /
answer / frames / target image). The training dataloader expects the interleaved
BAGEL schema: image_list / instruction_list / output_text_list / num_input_images,
plus a parquet_info.json. This script converts the former into the latter.

Flow:  download_spatial_datasets.py  ->  prepare_datasets.py  ->  train_*.sh

Usage:
  python scripts/prepare_datasets.py --task mvc       # MVC only
  python scripts/prepare_datasets.py --task pet       # PET only
  python scripts/prepare_datasets.py --task pt        # PT (already training-format: just index)
  python scripts/prepare_datasets.py                  # all

Each task reads data/training/<local_name>/ (the downloaded raw parquet) and writes
chunk_0.parquet + parquet_info.json into the same folder, ready for training.
"""
import os, glob, json, random, argparse
import pyarrow as pa
import pyarrow.parquet as pq

random.seed(0)

VLM_THINK = ("\nLet's think step by step to answer the question. For text-based thinking, "
             "enclose the process within <think> </think>, e.g. <think> thinking process here "
             "</think>. For visual thinking, enclose the content within <image_start> </image_end>, "
             "e.g. <image_start> thinking image here </image_end>. Finally conclude with the final "
             "answer wrapped in <answer></answer> tags, i.e.<answer> answer here </answer>.")

MVC_THINKS = ["Examining the scene from above to identify all objects.",
              "Let me consider the overlap between different camera views.",
              "I'll examine the top-down perspective to count the objects.",
              "Analyzing the different perspectives to avoid double-counting."]
PET_THINKS = ["I need to visualize the scene from the new perspective after moving to the marked position.",
              "Let me imagine what the view looks like from the target position.",
              "Visualizing the novel viewpoint to determine the spatial relation."]

DATA_ROOT = os.environ.get("DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training"))


def _img(v):
    """Extract raw image bytes from a HF Image cell (struct{bytes,path})."""
    return v["bytes"] if isinstance(v, dict) else v


def _write(out_dir, image_list, instr, out_text, num_in):
    os.makedirs(out_dir, exist_ok=True)
    tbl = pa.table({
        "image_list":       pa.array(image_list, type=pa.list_(pa.binary())),
        "instruction_list": pa.array(instr,      type=pa.list_(pa.string())),
        "output_text_list": pa.array(out_text,   type=pa.list_(pa.string())),
        "num_input_images": pa.array(num_in,     type=pa.int64()),
    })
    chunk = os.path.join(out_dir, "chunk_0.parquet")
    pq.write_table(tbl, chunk, row_group_size=256)
    pf = pq.ParquetFile(chunk)
    json.dump({chunk: {"num_row_groups": pf.metadata.num_row_groups, "num_rows": pf.metadata.num_rows}},
              open(os.path.join(out_dir, "parquet_info.json"), "w"))
    print(f"  wrote {chunk}: {tbl.num_rows} rows, {pf.metadata.num_row_groups} row groups")


def _raw_rows(local_name):
    d = os.path.join(DATA_ROOT, local_name)
    files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
    files = [f for f in files if "chunk_" not in os.path.basename(f)]
    for f in files:
        for r in pq.read_table(f).to_pylist():
            yield r


def prepare_mvc():
    print("[MVC] transform raw -> training format")
    IL, INS, OUT, NIN = [], [], [], []
    for r in _raw_rows("multi_view_counting"):
        frames = [_img(r[f"frame_{i}"]) for i in range(8) if r.get(f"frame_{i}") and _img(r[f"frame_{i}"])]
        tgt = _img(r.get("topdown_map")) if r.get("topdown_map") else None
        if not frames or not tgt:
            continue
        IL.append(frames + [tgt]); NIN.append(len(frames))
        INS.append([VLM_THINK + "\n" + str(r["question"])])
        OUT.append(["<think>" + random.choice(MVC_THINKS) + "</think><image_start>",
                    "<image_end><answer>" + str(r["answer"]) + "</answer>"])
    _write(os.path.join(DATA_ROOT, "multi_view_counting"), IL, INS, OUT, NIN)


def prepare_pet():
    print("[PET] transform raw -> training format")
    IL, INS, OUT, NIN = [], [], [], []
    for r in _raw_rows("perspective_with_sysprompt"):
        mi = _img(r.get("marked_image_no_arrow")) if r.get("marked_image_no_arrow") else None
        tgt = _img(r.get("new_perspective")) if r.get("new_perspective") else None
        if not mi or not tgt:
            continue
        IL.append([mi, tgt]); NIN.append(1)
        INS.append([VLM_THINK + "\n" + str(r.get("question_no_arrow") or r.get("question"))])
        OUT.append(["<think>" + random.choice(PET_THINKS) + "</think><image_start>",
                    "<image_end><answer>" + str(r["answer"]) + "</answer>"])
    _write(os.path.join(DATA_ROOT, "perspective_with_sysprompt"), IL, INS, OUT, NIN)


def prepare_pt():
    # PT released datasets are already in training schema; just (re)build parquet_info.json.
    print("[PT] already training-format: indexing parquet_info.json")
    d = os.path.join(DATA_ROOT, "path_tracing_with_sysprompt")
    chunks = sorted(glob.glob(os.path.join(d, "*.parquet")))
    info = {}
    for c in chunks:
        pf = pq.ParquetFile(c)
        info[c] = {"num_row_groups": pf.metadata.num_row_groups, "num_rows": pf.metadata.num_rows}
    json.dump(info, open(os.path.join(d, "parquet_info.json"), "w"))
    print(f"  indexed {len(chunks)} parquet file(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mvc", "pet", "pt", "all"], default="all")
    a = ap.parse_args()
    if a.task in ("mvc", "all"): prepare_mvc()
    if a.task in ("pet", "all"): prepare_pet()
    if a.task in ("pt",  "all"): prepare_pt()
    print("Done.")
