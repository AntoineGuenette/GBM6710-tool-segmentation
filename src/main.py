import argparse
import os

from segmentation import segment_tools
from analysis import compute_IoU

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
    GT_dir = os.path.join(data_dir, 'ground_truth')
    test_set_dir = os.path.join(data_dir, 'test_set')
    res_dir = os.path.join(data_dir, '..', 'res')

    # Iterate over all datasets (1 to 10)
    for i in range(1, 11):

        # Defin dataset-specific paths
        GT_frames_dir = os.path.join(GT_dir, f'instrument_dataset_{i}', 'BinarySegmentation')
        frames_dir = os.path.join(test_set_dir, f'instrument_dataset_{i}', 'left_frames')
        save_dir = os.path.join(res_dir, f'instrument_dataset_{i}')
        os.makedirs(save_dir, exist_ok=True) # Make directory if it does not exist

        # iterate over all frames
        for filename in sorted(os.listdir(frames_dir)):
            # Only consider PNGs
            if not filename.lower().endswith((".png")):
                continue
            
            # Define frame paths
            frame_path = os.path.join(frames_dir, filename)
            GT_path = os.path.join(GT_frames_dir, filename)

            # Ignore macOS metadata files (e.g., '._frame225.png')
            if os.path.basename(filename).startswith('._'):
                continue
            
            # Segment the frame
            segment_tools(frame_path, save_dir)

if __name__ == "__main__" :
    main()