import argparse
import os
import logging

from segmentation import segment_tools
from analysis import compute_IoU, compute_mean_IoU, append_dataset_result

logging.basicConfig(
    level=logging.INFO,  # temporary default, will be overridden
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentation pipeline on EndoVis2017 dataset"
    )
    parser.add_argument("--data-dir", type=str, help="Path to the EndoVis2017 folder")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO"],
        help="Logging level"
    )
    return parser.parse_args()

def main():
    # Extract arguments
    args = parse_args()
    data_dir = args.data_dir
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    logger.info(f"Starting segmentation pipeline with data_dir={data_dir} | log_level={args.log_level}")

    # Define other paths
    GT_dir = os.path.join(data_dir, 'ground_truth')
    test_set_dir = os.path.join(data_dir, 'test_set')
    res_dir = os.path.join(data_dir, '..', 'res')
    csv_path = os.path.join(res_dir, "metrics.csv")

    # Iterate over all datasets (1 to 10)
    for i in range(1, 11):

        logger.info(f"Processing dataset {i}")
        iou_list = []

        # Define dataset-specific paths
        GT_frames_dir = os.path.join(GT_dir, f'instrument_dataset_{i}', 'BinarySegmentation')
        frames_dir = os.path.join(test_set_dir, f'instrument_dataset_{i}', 'left_frames')
        save_dir = os.path.join(res_dir, f'instrument_dataset_{i}')
        os.makedirs(save_dir, exist_ok=True) # Make directory if it does not exist

        # terate over all frames
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
            
            logger.debug(f"Processing file: {filename}")

            # Segment the frame
            segment_tools(frame_path, save_dir)

            # Load masks
            computed_mask = None  # TODO: replace with actual loading of saved mask
            GT_mask = None        # TODO: replace with actual loading of GT mask

            # Compute IoU if masks are available
            if computed_mask is not None and GT_mask is not None:
                iou = compute_IoU(computed_mask, GT_mask)
                iou_list.append(iou)

        mean_iou = compute_mean_IoU(iou_list)
        logger.info(f"Dataset {i} - Mean IoU: {mean_iou:.4f}")

        append_dataset_result(csv_path, i, mean_iou)

if __name__ == "__main__" :
    main()