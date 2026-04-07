import argparse
import os
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segmentation import segment_tools, crop_image
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
    debug_vis = (log_level == logging.DEBUG)

    # Set level for root logger AND all module loggers
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    # Set log level to warning for python librairies
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info(f"Starting segmentation pipeline with data_dir={data_dir} | log_level={args.log_level}")

    # Define other paths
    GT_dir = os.path.join(data_dir, 'ground_truth')
    test_set_dir = os.path.join(data_dir, 'test_set')
    res_dir = os.path.join(data_dir, '..', 'res')
    csv_path = os.path.join(res_dir, "metrics.csv")

    # Store global IoU results for best/worst/median selection
    all_results = []

    # Iterate over all datasets (1 to 10)
    for i in range(1, 2):

        logger.info(f"-> PROCESSING DATASET {i} <-")
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
            mask_path = os.path.join(save_dir, 'binary_segmentations', 'bin_' + filename)
            cropped_path = os.path.join(save_dir, 'cropped_images', 'cropped_' + filename)

            # Ignore macOS metadata files (e.g., '._frame225.png')
            if os.path.basename(filename).startswith('._'):
                continue
            
            logger.debug(f"-> PROCESSING FILE: {filename} <-")

            # Segment the frame
            computed_mask, prob_map, img_crop = segment_tools(frame_path, save_dir, debug=debug_vis)

            # Load GT (as binary)
            if os.path.exists(mask_path) and os.path.exists(GT_path):
                GT_mask = cv2.imread(GT_path, cv2.IMREAD_GRAYSCALE) # Open
                GT_mask = crop_image(GT_mask, 328, 37, 1264, 1010) # Crop

                # Concert to binary in uint8 (0/1)
                computed_mask = (computed_mask > 0).astype(np.uint8)
                GT_mask = (GT_mask > 0).astype(np.uint8)
            else:
                logger.warning(f"Missing mask for {filename}, skipping IoU computation")
                computed_mask = None
                GT_mask = None

            if computed_mask is not None and GT_mask is not None:
                # Compute IoU
                iou = compute_IoU(computed_mask, GT_mask)
                iou_list.append(iou)

                all_results.append({
                    "iou": iou,
                    "filename": filename,
                    "img_crop": img_crop,
                    "GT_mask": GT_mask,
                    "prob_map": prob_map,
                    "computed_mask": computed_mask,
                    "save_dir": save_dir
                })

                # Save complete qualitative visualization
                fig, axes = plt.subplots(2, 2, figsize=(8, 8))

                axes[0,0].imshow(img_crop)
                axes[0,0].set_title("Original Image")
                axes[0,0].axis("off")

                axes[0,1].imshow(GT_mask, cmap="gray")
                axes[0,1].set_title("Ground Truth")
                axes[0,1].axis("off")

                axes[1,0].imshow(prob_map, cmap="gray")
                axes[1,0].set_title("Probability Map")
                axes[1,0].axis("off")

                axes[1,1].imshow(computed_mask, cmap="gray")
                axes[1,1].set_title("Final Mask")
                axes[1,1].axis("off")

                plt.suptitle(f"{filename}\nIoU: {iou:.4f}")
                plt.tight_layout()

                qual_res_file_name = 'QualRes_' + filename
                qual_res_file_path = os.path.join(save_dir, 'qualitative_results', qual_res_file_name)
                os.makedirs(os.path.dirname(qual_res_file_path), exist_ok=True)
                plt.savefig(qual_res_file_path, dpi=300)

        # Compute mean IoU
        mean_iou = compute_mean_IoU(iou_list)
        logger.info(f"Dataset {i} - Mean IoU: {mean_iou:.4f}")

        # Save mean IoUs in a CSV file
        append_dataset_result(csv_path, i, mean_iou)

    # Global qualitative selection (min, max, median IoU)
    if len(all_results) > 0:
        # Sort by IoU
        all_results_sorted = sorted(all_results, key=lambda x: x["iou"])

        min_sample = all_results_sorted[0]
        max_sample = all_results_sorted[-1]
        median_sample = all_results_sorted[len(all_results_sorted)//2]

        selected = {
            "min": min_sample,
            "median": median_sample,
            "max": max_sample
        }

        global_vis_dir = os.path.join(res_dir, "global_qualitative")
        os.makedirs(global_vis_dir, exist_ok=True)

        for key, sample in selected.items():
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))

            axes[0,0].imshow(sample["img_crop"])
            axes[0,0].set_title("Original Image")
            axes[0,0].axis("off")

            axes[0,1].imshow(sample["GT_mask"], cmap="gray")
            axes[0,1].set_title("Ground Truth")
            axes[0,1].axis("off")

            axes[1,0].imshow(sample["prob_map"], cmap="gray")
            axes[1,0].set_title("Probability Map")
            axes[1,0].axis("off")

            axes[1,1].imshow(sample["computed_mask"], cmap="gray")
            axes[1,1].set_title("Final Mask")
            axes[1,1].axis("off")

            plt.suptitle(f"{sample['filename']} ({key})\nIoU: {sample['iou']:.4f}")
            plt.tight_layout()

            out_path = os.path.join(global_vis_dir, f"{key}_IoU_{sample['iou']:.4f}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()

        logger.info("Saved global qualitative samples (min, median, max IoU)")

if __name__ == "__main__" :
    main()