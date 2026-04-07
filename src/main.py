import argparse
import os

from segmentation import segment_tools

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentation pipeline on EndoVis2017 dataset"
    )
    parser.add_argument("--data-dir", type=str, help="Path to the EndoVis2017 folder")
    return parser.parse_args()

def main():
    # Extract arguments
    args = parse_args()
    data_dir = args.data_dir

    # Define other paths
    repo_dir = os.path.join(data_dir, '..')
    res_dir = os.path.join(repo_dir, 'res')

    # Segment all images in directory
    for img in data_dir:
        pass

if __name__ == "__main__" :
    main()