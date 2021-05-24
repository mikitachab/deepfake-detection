import argparse
import os
import shutil
from tqdm import tqdm

def main(args):
    with open("exp_files.txt") as f:
        exp_files = [file.strip() for file in f.readlines()]

    for file in tqdm(exp_files):
        source = os.path.join(args.source, file)
        dest = os.path.join(args.dest, file)
        shutil.copyfile(source, dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str)
    parser.add_argument("--dest", "-d", type=str)
    args = parser.parse_args()
    main(args)
