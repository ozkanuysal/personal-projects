from argparse import ArgumentParser
from collections import defaultdict
from datasets import load_from_disk, Dataset
from pathlib import Path
from pprint import pprint
import gc
import numpy as np
import os
import psutil
import sys
import torchvision.transforms as transforms
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = "./data/cm4"

DATASET_PATH = f"{DATA_ROOT}/cm4-10000-v0.1"

parser = ArgumentParser()

parser.add_argument("--dataset_name_or_path", required=True, type=str, help="source dataset_name_or_path")
parser.add_argument("--target_path", required=False, default="output", type=str, help="path to where to unpack")
parser.add_argument("--ids", required=False, default="0", type=str, help="which ids to extract. example: 1,2,5-7,10")
args = parser.parse_args()

def list2range(s):
    return sum(((list(range(*[int(j) + k for k,j in enumerate(i.split('-'))]))
         if '-' in i else [int(i)]) for i in s.split(',')), [])

def unpack(args, idx, row):
    path = f"{args.target_path}/{idx}"
    Path(path).mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(row["images"]):
        basename = f"{path}/images_{i:02d}"
        ext = "null" if img is None else "jpg"
        file = f"{basename}.{ext}"
        with open(file, "w") as fh:
            if img is not None:
                img.save(fh, 'jpeg')
    for i, txt in enumerate(row["texts"]):
        basename = f"{path}/texts_{i:02d}"
        ext = "null" if txt is None else "txt"
        file = f"{basename}.{ext}"
        with open(file, "w") as fh:
            if txt is not None:
                fh.write(txt)

def dump_example_shapes(idx, row):
    imgs = defaultdict(int)
    for img in row["images"]:
        if img is None:
            imgs["0"] += 1
        else:
            shape = "x".join(map(str, img.size))
            imgs[shape] += 1
    imgs_summary = ", ".join([f"{v} {k}" for k,v in sorted(imgs.items(), key=lambda x: int(x[0].split("x")[0]))])

    txts = defaultdict(int)
    for txt in row["texts"]:
        if txt is None:
            txts[0] += 1
        else:
            shape = len(txt)
            txts[shape] += 1
    txts_summary = ", ".join([f"{v} {k}" for k,v in sorted(txts.items(), key=lambda x: int(x[0]))])

    print(f"\nrec{idx}: {len(row['images'])} pairs with {len(row['images'])-imgs['0']} images, {len(row['texts'])-txts[0]} texts")
    print(f"- img: {imgs_summary}")
    print(f"- txt: {txts_summary}")

ids_range = list2range(args.ids)
ds = load_from_disk(args.dataset_name_or_path)
for idx, id in enumerate(ids_range):
    unpack(args, id, ds[id])
    dump_example_shapes(id, ds[id])
    ds.info.write_to_directory(args.target_path)