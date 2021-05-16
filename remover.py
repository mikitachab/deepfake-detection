import argparse
import os
import json
import random

def remove(metadata):
    for _ in range(700):
        files = os.listdir("preprocessed_data")
        file = random.choice(files)
        if metadata[file]["label"] == "FAKE":
            print("removing", file)
            os.remove(os.path.join("preprocessed_data", file))

def main(args):
    with open("file_to_stay.txt") as f:
        file_to_stay = f.read().splitlines()

    for file in os.listdir(args.dir):
        if file not in file_to_stay:
            print("removing", file)
            os.remove(os.path.join(args.dir, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str)
    args = parser.parse_args()
    main(args)
