"""Prepare calibration/evaluation datasets used by ReCalKV.

Downloads WikiText-2, PTB and C4 and stores them on disk using the exact
directory layout that the loaders in ``palu`` and ``run_ppl_eval.py`` expect::

    <data_root>/wikitext/traindata   (column: text)
    <data_root>/wikitext/testdata    (column: text)
    <data_root>/ptb/traindata        (column: sentence)
    <data_root>/ptb/testdata         (column: sentence)
    <data_root>/c4/traindata         (column: text)
    <data_root>/c4/valdata           (column: text)

Usage:
    export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
    python prepare_data.py --data_root ./data
"""
import argparse
import os

from datasets import load_dataset


def _save(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.save_to_disk(path)
    print(f"[saved] {path}  ({len(ds)} rows, columns={ds.column_names})")


def prepare_wikitext(data_root):
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    _save(train, os.path.join(data_root, "wikitext", "traindata"))
    _save(test, os.path.join(data_root, "wikitext", "testdata"))


def prepare_ptb(data_root):
    train = load_dataset("ptb_text_only", "penn_treebank", split="train", trust_remote_code=True)
    test = load_dataset("ptb_text_only", "penn_treebank", split="test", trust_remote_code=True)
    _save(train, os.path.join(data_root, "ptb", "traindata"))
    _save(test, os.path.join(data_root, "ptb", "testdata"))


def prepare_c4(data_root):
    train = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    val = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    _save(train, os.path.join(data_root, "c4", "traindata"))
    _save(val, os.path.join(data_root, "c4", "valdata"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--datasets",
        type=str,
        default="wikitext,ptb,c4",
        help="Comma-separated subset of {wikitext,ptb,c4} to prepare.",
    )
    args = parser.parse_args()

    todo = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if "wikitext" in todo:
        prepare_wikitext(args.data_root)
    if "ptb" in todo:
        prepare_ptb(args.data_root)
    if "c4" in todo:
        prepare_c4(args.data_root)
    print("All done.")


if __name__ == "__main__":
    main()
