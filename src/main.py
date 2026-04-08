import argparse
import os
import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segmentation import segment_tools, crop_image
from analysis import compute_IoU, compute_mean_IoU, append_dataset_result, append_global_mean, save_all_ious, load_all_ious
from figures import plot_qualitative_results, plot_bar_comparison, plot_violin_iou

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
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip segmentation and IoU computation, load from CSV instead"
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
    mean_ious_csv_path = os.path.join(res_dir, "mean_ious.csv")
    all_ious_csv_path = os.path.join(res_dir, "all_ious.csv")

    if args.skip_analysis:
        all_ious_per_dataset = load_all_ious(all_ious_csv_path)
        dataset_ids = list(all_ious_per_dataset.keys())
        mean_ious = [compute_mean_IoU(all_ious_per_dataset[d]) for d in dataset_ids]
    else:
        # Store global IoU results for best/worst/median selection
        all_results = []

        all_ious_per_dataset = {}
        mean_ious = []

        # Iterate over all datasets (1 to 10)
        for i in range(1, 11):

            logger.info(f"-> PROCESSING DATASET {i} <-")
            iou_list = []

            # Define dataset-specific paths
            GT_frames_dir = os.path.join(GT_dir, f'instrument_dataset_{i}', 'BinarySegmentation')
            frames_dir = os.path.join(test_set_dir, f'instrument_dataset_{i}', 'left_frames')
            save_dir = os.path.join(res_dir, f'instrument_dataset_{i}')
            os.makedirs(save_dir, exist_ok=True) # Make directory if it does not exist

            # Iterate over all frames
            for filename in sorted(os.listdir(frames_dir)):
                # Only consider PNGs
                if not filename.lower().endswith((".png")):
                    continue
                
                # Define frame paths
                frame_path = os.path.join(frames_dir, filename)
                GT_path = os.path.join(GT_frames_dir, filename)
                mask_path = os.path.join(save_dir, 'binary_segmentations', 'bin_' + filename)
                cropped_path = os.path.join(save_dir, 'cropped_images', 'cropped_' + filename)
                qual_res_file_name = 'QualRes_' + filename
                qual_res_file_path = os.path.join(save_dir, 'qualitative_results', qual_res_file_name)

                # Ignore macOS metadata files (e.g., '._frame225.png')
                if os.path.basename(filename).startswith('._'):
                    continue

                # Check if qualitative result already exists (i.e., output image)
                output_image_path = qual_res_file_path
                pred_mask_path = mask_path
                gt_mask_path = GT_path

                if os.path.exists(output_image_path) and os.path.exists(pred_mask_path):
                    
                    logger.debug(f"Skipping existing image: {output_image_path}")
                    # OPTIONAL: still compute IoU if masks are available
                    if os.path.exists(pred_mask_path) and os.path.exists(gt_mask_path):
                        # Load predicted mask
                        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                        # Crop GT mask as before
                        gt_mask = crop_image(gt_mask, 328, 37, 1264, 1010)
                        # Convert to binary
                        pred_mask = (pred_mask > 0).astype(np.uint8)
                        gt_mask = (gt_mask > 0).astype(np.uint8)
                        iou = compute_IoU(pred_mask, gt_mask)
                        iou_list.append(iou)
                        # Store for global qualitative selection
                        all_results.append({
                            "iou": iou,
                            "filename": filename,
                            "img_crop": None,
                            "GT_mask": gt_mask,
                            "prob_map": None,
                            "computed_mask": pred_mask,
                            "save_dir": save_dir
                        })
                    continue
                else:
                    logger.debug(f"-> PROCESSING FILE: {filename} <-")

                    # Segment the frame
                    computed_mask, prob_map, img_crop = segment_tools(frame_path, save_dir, debug=debug_vis)

                    # Save predicted mask (already saved in segment_tools)
                    # Also save GT mask if exists
                    if os.path.exists(gt_mask_path):
                        GT_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE) # Open
                        GT_mask = crop_image(GT_mask, 328, 37, 1264, 1010) # Crop
                        # Convert to binary in uint8 (0/1)
                        computed_mask_bin = (computed_mask > 0).astype(np.uint8)
                        GT_mask_bin = (GT_mask > 0).astype(np.uint8)
                    else:
                        logger.warning(f"Missing mask for {filename}, skipping IoU computation")
                        computed_mask_bin = None
                        GT_mask_bin = None

                    if computed_mask_bin is not None and GT_mask_bin is not None:
                        # Compute IoU
                        iou = compute_IoU(computed_mask_bin, GT_mask_bin)
                        iou_list.append(iou)

                        all_results.append({
                            "iou": iou,
                            "filename": filename,
                            "img_crop": img_crop,
                            "GT_mask": GT_mask_bin,
                            "prob_map": prob_map,
                            "computed_mask": computed_mask_bin,
                            "save_dir": save_dir
                        })

                        plot_qualitative_results(
                            img_crop,
                            GT_mask_bin,
                            prob_map,
                            computed_mask_bin,
                            title=f"{filename}\nIoU: {iou:.4f}",
                            save_path=qual_res_file_path
                        )

            # Compute mean IoU
            mean_iou = compute_mean_IoU(iou_list)
            all_ious_per_dataset[i] = iou_list
            mean_ious.append(mean_iou)
            logger.info(f"Dataset {i} - Mean IoU: {mean_iou:.4f}")

            # Save mean IoUs in a CSV file
            append_dataset_result(mean_ious_csv_path, i, mean_iou)

        # Save all IoUs to CSV
        save_all_ious(all_ious_csv_path, all_ious_per_dataset)

    dataset_ids = list(all_ious_per_dataset.keys())

    # Compute global mean of means
    global_mean_iou = float(np.mean(mean_ious)) if len(mean_ious) > 0 else 0.0
    logger.info(f"Global mean IoU (mean of means): {global_mean_iou:.4f}")

    # Save to CSV
    append_global_mean(mean_ious_csv_path, global_mean_iou)

    # Global qualitative selection (top 3 min, median, max IoU)
    if not args.skip_analysis and len(all_results) > 0:
        # Sort by IoU
        all_results_sorted = sorted(all_results, key=lambda x: x["iou"])
        n = len(all_results_sorted)

        # Select samples

        # Exclude IoU == 0 for minimum samples
        non_zero_results = [r for r in all_results_sorted if r["iou"] > 0]

        if len(non_zero_results) >= 3:
            min_samples = non_zero_results[:3]
        else:
            min_samples = non_zero_results  # fallback if not enough samples

        max_samples = all_results_sorted[-3:]

        median_start = max(0, n // 2 - 1)
        median_samples = all_results_sorted[median_start:median_start + 3]

        selected_groups = {
            "minimum": min_samples,
            "médiane": median_samples,
            "maximum": max_samples
        }

        global_vis_dir = os.path.join(res_dir, "global_qualitative")
        os.makedirs(global_vis_dir, exist_ok=True)

        for key, samples in selected_groups.items():
            for idx, sample in enumerate(samples):
                out_path = os.path.join(
                    global_vis_dir,
                    f"{key}_{idx+1}_IoU_{sample['iou']:.4f}.png"
                )

                plot_qualitative_results(
                    sample["img_crop"],
                    sample["GT_mask"],
                    sample["prob_map"],
                    sample["computed_mask"],
                    title=f"{sample['filename']} ({key} #{idx+1})\nIoU: {sample['iou']:.4f}",
                    save_path=out_path
                )

        logger.info("Saved global qualitative samples (top 3 min, median, max IoU)")

    # Quantitative plots

    if args.skip_analysis:
        if not os.path.exists(all_ious_csv_path):
            raise FileNotFoundError(
                f"{all_ious_csv_path} not found. Run without --skip-analysis first."
            )

    # Article IoUs (from provided table)
    article_ious = [0.337, 0.289, 0.483, 0.678, 0.219, 0.619, 0.325, 0.506, 0.377, 0.603]
    article_global_mean = 0.461

    # Bar comparison
    bar_plot_path = os.path.join(res_dir, "mean_iou_comparison.png")
    plot_bar_comparison(
        mean_ious + [global_mean_iou],
        article_ious[:len(mean_ious)] + [article_global_mean],
        bar_plot_path
    )

    # Violin plot
    violin_plot_path = os.path.join(res_dir, "iou_violin_plot.png")
    plot_violin_iou(all_ious_per_dataset, dataset_ids, violin_plot_path)

    logger.info("Saved quantitative plots (bar + violin)")

if __name__ == "__main__" :
    main()