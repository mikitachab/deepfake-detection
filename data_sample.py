import argparse
import json
import os
import shutil


def all_by_label(metadata, files, label):
    return [
        file for file in files if metadata[file]["label"] == label
    ]


def main(args):
    os.makedirs(args.dest, exist_ok=True)
    with open(args.meta) as f:
        metadata = json.load(f)

    files = [f for f in os.listdir(args.source) if f.endswith("mp4")]
    for label in ["REAL", "FAKE"]:
        for file in all_by_label(metadata, files, label)[:args.size]:
            shutil.copyfile(os.path.join(args.source, file), os.path.join(args.dest, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--meta", "-m", type=str)
    parser.add_argument("--size", "-s", type=int)
    args = parser.parse_args()
    main(args)
