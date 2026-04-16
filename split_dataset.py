import argparse
import csv
import os
import random
from typing import List, Tuple


def split_indices(num_items: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)

    train_count = int(num_items * train_ratio)
    train_count = max(1, min(train_count, num_items - 1))

    train_indices = sorted(indices[:train_count])
    val_indices = sorted(indices[train_count:])
    return train_indices, val_indices


def split_txt(input_path: str, train_output: str, val_output: str, train_ratio: float, seed: int) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    train_indices, val_indices = split_indices(len(lines), train_ratio, seed)

    with open(train_output, "w", encoding="utf-8") as f:
        for idx in train_indices:
            f.write(lines[idx])

    with open(val_output, "w", encoding="utf-8") as f:
        for idx in val_indices:
            f.write(lines[idx])


def split_csv(input_path: str, train_output: str, val_output: str, train_ratio: float, seed: int) -> None:
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {input_path}")

    header, data_rows = rows[0], rows[1:]
    train_indices, val_indices = split_indices(len(data_rows), train_ratio, seed)

    with open(train_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in train_indices:
            writer.writerow(data_rows[idx])

    with open(val_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx in val_indices:
            writer.writerow(data_rows[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a dataset list file into train/val subsets.")
    parser.add_argument("--input", required=True, help="Input .txt or .csv file")
    parser.add_argument("--train-output", required=True, help="Output path for the train split")
    parser.add_argument("--val-output", required=True, help="Output path for the validation split")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio, default 0.8")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".txt":
        split_txt(args.input, args.train_output, args.val_output, args.train_ratio, args.seed)
    elif ext == ".csv":
        split_csv(args.input, args.train_output, args.val_output, args.train_ratio, args.seed)
    else:
        raise ValueError("Only .txt and .csv inputs are supported.")

    print(f"Created train split: {args.train_output}")
    print(f"Created val split: {args.val_output}")


if __name__ == "__main__":
    main()
