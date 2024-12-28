import argparse
from collections import defaultdict
from pathlib import Path
from typing import List

import gc
import numpy as np
import os
import psutil
from datasets import load_from_disk, Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


DATA_ROOT = "./data/cm4"
DATASET_PATH = f"{DATA_ROOT}/cm4-10000-v0.1"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unpack dataset images and texts.")
    parser.add_argument(
        "--dataset_name_or_path",
        required=True,
        type=str,
        help="Source dataset name or path.",
    )
    parser.add_argument(
        "--target_path",
        default="output",
        type=str,
        help="Path to unpack the dataset.",
    )
    parser.add_argument(
        "--ids",
        default="0",
        type=str,
        help="IDs to extract (e.g., 1,2,5-7,10).",
    )
    return parser.parse_args()


def list_to_range(s: str) -> List[int]:
    ranges = []
    for part in s.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ranges.extend(range(start, end + 1))
        else:
            ranges.append(int(part))
    return ranges


def unpack_image(target_dir: Path, idx: int, img, img_index: int) -> None:
    image_path = target_dir / f"images_{img_index:02d}.jpg" if img else target_dir / f"images_{img_index:02d}.null"
    if img:
        img.save(image_path, 'JPEG')


def unpack_text(target_dir: Path, idx: int, txt: Optional[str], txt_index: int) -> None:
    text_path = target_dir / f"texts_{txt_index:02d}.txt" if txt else target_dir / f"texts_{txt_index:02d}.null"
    if txt:
        with open(text_path, "w", encoding='utf-8') as fh:
            fh.write(txt)


def unpack(args: argparse.Namespace, idx: int, row) -> None:
    path = Path(args.target_path) / str(idx)
    path.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(row.get("images", [])):
        unpack_image(path, idx, img, i)

    for i, txt in enumerate(row.get("texts", [])):
        unpack_text(path, idx, txt, i)


def dump_example_shapes(idx: int, row) -> None:
    imgs = defaultdict(int)
    for img in row.get("images", []):
        if img is None:
            imgs["0"] += 1
        else:
            shape = "x".join(map(str, img.size))
            imgs[shape] += 1
    imgs_summary = ", ".join([f"{v} {k}" for k, v in sorted(imgs.items(), key=lambda x: int(x[0].split('x')[0]))])

    txts = defaultdict(int)
    for txt in row.get("texts", []):
        if txt is None:
            txts["0"] += 1
        else:
            txts[str(len(txt))] += 1
    txts_summary = ", ".join([f"{v} {k}" for k, v in sorted(txts.items(), key=lambda x: int(x[0]))])

    print(f"\nrec{idx}: {len(row.get('images', []))} pairs with {len(row.get('images', [])) - imgs['0']} images, {len(row.get('texts', [])) - txts['0']} texts")
    print(f"- img: {imgs_summary}")
    print(f"- txt: {txts_summary}")


def main() -> None:
    args = parse_arguments()
    ids_range = list_to_range(args.ids)
    dataset = load_from_disk(args.dataset_name_or_path)

    for id in ids_range:
        unpack(args, id, dataset[id])
        dump_example_shapes(id, dataset[id])

    dataset.info.write_to_directory(args.target_path)


if __name__ == "__main__":
    main()