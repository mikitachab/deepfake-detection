import os
import json
from collections import defaultdict


def main():
     
    with open("metadata.json") as f:
        metadata = json.load(f)
    
    d = defaultdict(int)

    for file in [f for f in os.listdir("preprocessed_data") if f.endswith("mp4")]:
        d[metadata[file]["label"]] += 1
    
    print(d)
    
if __name__ == "__main__":
    main()