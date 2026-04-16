import argparse
import os
import logging
import cv2
import numpy as np
import shutil

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
    all_ious_csv_path_border = os.path.join(res_dir, "all_ious_border.csv")
    all_ious_csv_path_valid = os.path.join(res_dir, "all_ious_valid.csv")

    if args.skip_analysis:
        # For skip-analysis, load both border and valid csvs
        all_ious_border, all_ious_valid = load_all_ious(all_ious_csv_path_border)
        all_ious_per_dataset = {d: {"border": all_ious_border[d], "valid_region": all_ious_valid[d]} for d in all_ious_border}
        dataset_ids = list(all_ious_per_dataset.keys())
        mean_ious = [{"border": compute_mean_IoU(all_ious_per_dataset[d]["border"]),
                      "valid_region": compute_mean_IoU(all_ious_per_dataset[d]["valid_region"])} for d in dataset_ids]
    else:
        # Store global IoU results for best/worst/median selection
        all_results = []

        all_ious_per_dataset = {}
        mean_ious = []

        # Iterate over all datasets (1 to 10)
        for i in range(1, 2):

            logger.info(f"-> PROCESSING DATASET {i} <-")
            iou_list_border = []
            iou_list_valid = []

            # Define dataset-specific paths
            GT_frames_dir = os.path.join(GT_dir, f'instrument_dataset_{i}', 'BinarySegmentation')
            frames_dir = os.path.join(test_set_dir, f'instrument_dataset_{i}', 'left_frames')
            dataset_dir = os.path.join(res_dir, f'instrument_dataset_{i}')
            save_dir_valid = os.path.join(dataset_dir, 'valid')
            save_dir_border = os.path.join(dataset_dir, 'border')
            os.makedirs(dataset_dir, exist_ok=True)

            # Iterate over all frames
            for filename in sorted(os.listdir(frames_dir)):
                # Only consider PNGs
                if not filename.lower().endswith((".png")):
                    continue
                
                # Define frame paths
                frame_path = os.path.join(frames_dir, filename)
                GT_path = os.path.join(GT_frames_dir, filename)
                mask_path_valid = os.path.join(save_dir_valid, 'binary_segmentations', 'bin_' + filename)
                mask_path_border = os.path.join(save_dir_border, 'binary_segmentations', 'bin_' + filename)
                cropped_path = os.path.join(dataset_dir, 'cropped_images', 'cropped_' + filename)
                qual_res_file_name = 'QualRes_' + filename
                qual_res_file_path_valid = os.path.join(save_dir_valid, 'qualitative_results', qual_res_file_name)
                qual_res_file_path_border = os.path.join(save_dir_border, 'qualitative_results', qual_res_file_name)

                # Ignore macOS metadata files (e.g., '._frame225.png')
                if os.path.basename(filename).startswith('._'):
                    continue

                # Check if qualitative result already exists (i.e., output image)
                output_image_path = qual_res_file_path_valid
                pred_mask_path = mask_path_valid
                gt_mask_path = GT_path

                if os.path.exists(output_image_path) and os.path.exists(pred_mask_path):
                    logger.debug(f"Skipping existing image: {output_image_path}")

                    # --- Reload original image and crop ---
                    original_img = cv2.imread(frame_path)
                    if original_img is None:
                        raise ValueError(f"Failed to reload image: {frame_path}")

                    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    img_crop = crop_image(original_img_rgb, 328, 37, 1264, 1010)

                    if img_crop is None:
                        raise ValueError("img_crop is None after reload in skip mode")

                    # --- Load predicted and GT masks ---
                    # Load VALID mask
                    pred_mask_valid = cv2.imread(mask_path_valid, cv2.IMREAD_GRAYSCALE)
                    pred_mask_valid = (pred_mask_valid > 0).astype(np.uint8)

                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    if pred_mask_valid is None or gt_mask is None:
                        raise ValueError("Failed to load masks in skip mode")

                    # Crop GT mask
                    gt_mask = crop_image(gt_mask, 328, 37, 1264, 1010)
                    gt_mask = (gt_mask > 0).astype(np.uint8)

                    iou_valid = compute_IoU(pred_mask_valid, gt_mask)
                    iou_list_valid.append(iou_valid)

                    # Load BORDER mask
                    pred_mask_border = cv2.imread(mask_path_border, cv2.IMREAD_GRAYSCALE)
                    pred_mask_border = (pred_mask_border > 0).astype(np.uint8)

                    iou_border = compute_IoU(pred_mask_border, gt_mask)
                    iou_list_border.append(iou_border)

                    prob_map_path = os.path.join(dataset_dir, 'prob_maps', 'prob_' + filename)

                    # Store for global qualitative selection
                    all_results.append({
                        "iou_valid": iou_valid,
                        "iou_border": iou_border,
                        "filename": filename,
                        "frame_path": frame_path,
                        "gt_path": gt_mask_path,
                        "pred_mask_path": mask_path_valid,
                        "prob_map_path": prob_map_path,
                        "save_dir": dataset_dir
                    })

                    continue
                else:
                    logger.debug(f"-> PROCESSING FILE: {filename} <-")

                    # Segment the frame with border method
                    computed_mask_border, prob_map_border, img_crop_border = segment_tools(frame_path, dataset_dir, save_subdir='border', debug=debug_vis, method="border")
                    # Segment the frame with valid_region method
                    computed_mask_valid, prob_map_valid, img_crop_valid = segment_tools(frame_path, dataset_dir, save_subdir='valid', debug=debug_vis, method="valid_region")

                    # Save predicted masks (already saved in segment_tools)
                    # Also save GT mask if exists
                    if os.path.exists(gt_mask_path):
                        GT_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE) # Open
                        GT_mask = crop_image(GT_mask, 328, 37, 1264, 1010) # Crop
                        # Convert to binary in uint8 (0/1)
                        computed_mask_bin_border = (computed_mask_border > 0).astype(np.uint8)
                        computed_mask_bin_valid = (computed_mask_valid > 0).astype(np.uint8)
                        GT_mask_bin = (GT_mask > 0).astype(np.uint8)
                    else:
                        logger.warning(f"Missing mask for {filename}, skipping IoU computation")
                        computed_mask_bin_border = None
                        computed_mask_bin_valid = None
                        GT_mask_bin = None

                    if computed_mask_bin_border is not None and GT_mask_bin is not None:
                        # Compute IoU border
                        iou_border = compute_IoU(computed_mask_bin_border, GT_mask_bin)
                        iou_list_border.append(iou_border)
                    else:
                        iou_border = None

                    if computed_mask_bin_valid is not None and GT_mask_bin is not None:
                        # Compute IoU valid region
                        iou_valid = compute_IoU(computed_mask_bin_valid, GT_mask_bin)
                        iou_list_valid.append(iou_valid)
                    else:
                        iou_valid = None

                    prob_map_path = os.path.join(dataset_dir, 'prob_maps', 'prob_' + filename)

                    all_results.append({
                        "iou_valid": iou_valid,
                        "iou_border": iou_border,
                        "filename": filename,
                        "frame_path": frame_path,
                        "gt_path": gt_mask_path,
                        "pred_mask_path": mask_path_valid,
                        "prob_map_path": prob_map_path,
                        "save_dir": dataset_dir
                    })

                    # Qualitative result for VALID method
                    plot_qualitative_results(
                        img_crop_valid,
                        GT_mask_bin,
                        prob_map_valid,
                        computed_mask_bin_valid,
                        title=f"{filename}\nIoU valid: {iou_valid:.4f}",
                        save_path=qual_res_file_path_valid
                    )

                    # Qualitative result for BORDER method
                    plot_qualitative_results(
                        img_crop_border,
                        GT_mask_bin,
                        prob_map_border,
                        computed_mask_bin_border,
                        title=f"{filename}\nIoU border: {iou_border:.4f}",
                        save_path=qual_res_file_path_border
                    )

            # Compute mean IoU for both methods
            mean_iou_border = compute_mean_IoU(iou_list_border)
            mean_iou_valid = compute_mean_IoU(iou_list_valid)
            all_ious_per_dataset[i] = {
                "border": iou_list_border,
                "valid_region": iou_list_valid
            }
            mean_ious.append({
                "border": mean_iou_border,
                "valid_region": mean_iou_valid
            })
            logger.info(f"Dataset {i} - Mean IoU Border: {mean_iou_border:.4f} Valid Region: {mean_iou_valid:.4f}")

            # Save mean IoUs in a CSV file with two columns
            file_exists = os.path.isfile(mean_ious_csv_path)
            with open(mean_ious_csv_path, 'a', newline='') as f:
                import csv
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["dataset_id", "mean_IoU_border", "mean_IoU_valid_region"])

                writer.writerow([i, f"{mean_iou_border:.6f}", f"{mean_iou_valid:.6f}"])

        # Save all IoUs to CSV
        # Save separate CSVs for border and valid
        flattened_ious_border = {}
        flattened_ious_valid = {}
        for dataset_id, iou_dict in all_ious_per_dataset.items():
            flattened_ious_border[dataset_id] = iou_dict["border"]
            flattened_ious_valid[dataset_id] = iou_dict["valid_region"]

        save_all_ious(all_ious_csv_path_border, flattened_ious_border, {d: [] for d in flattened_ious_border})
        save_all_ious(all_ious_csv_path_valid, {d: [] for d in flattened_ious_valid}, flattened_ious_valid)

    dataset_ids = list(all_ious_per_dataset.keys())

    # Compute global mean of means for both methods
    global_mean_iou_border = float(np.mean([m["border"] for m in mean_ious])) if len(mean_ious) > 0 else 0.0
    global_mean_iou_valid = float(np.mean([m["valid_region"] for m in mean_ious])) if len(mean_ious) > 0 else 0.0
    logger.info(f"Global mean IoU Border (mean of means): {global_mean_iou_border:.4f}")
    logger.info(f"Global mean IoU Valid Region (mean of means): {global_mean_iou_valid:.4f}")

    # Save to CSV global means
    with open(mean_ious_csv_path, 'a', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["global_mean", f"{global_mean_iou_border:.6f}", f"{global_mean_iou_valid:.6f}"])

    # Global qualitative selection for BOTH methods (border + valid)
    if len(all_ious_per_dataset) > 0:

        global_vis_dir = os.path.join(res_dir, "global_qualitative")
        border_dir = os.path.join(global_vis_dir, "border")
        valid_dir = os.path.join(global_vis_dir, "valid")
        os.makedirs(border_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

        for method in ["border", "valid_region"]:

            # Select correct directory and IoU key
            if method == "border":
                save_dir_method = border_dir
                iou_key = "border"
            else:
                save_dir_method = valid_dir
                iou_key = "valid_region"

            # Build flat list of results with IoU and metadata
            method_results = []
            for d in all_ious_per_dataset:
                for idx, iou in enumerate(all_ious_per_dataset[d][iou_key]):
                    method_results.append({
                        "iou": iou,
                        "dataset_id": d,
                        "index": idx
                    })

            if len(method_results) == 0:
                continue

            # Sort by IoU
            method_results_sorted = sorted(method_results, key=lambda x: x["iou"])
            n = len(method_results_sorted)

            # Exclude IoU == 0 for minimum samples
            non_zero_results = [r for r in method_results_sorted if r["iou"] > 0]

            if len(non_zero_results) >= 3:
                min_samples = non_zero_results[:3]
            else:
                min_samples = non_zero_results

            max_samples = method_results_sorted[-3:]

            median_start = max(0, n // 2 - 1)
            median_samples = method_results_sorted[median_start:median_start + 3]

            selected_groups = {
                "minimum": min_samples,
                "mediane": median_samples,
                "maximum": max_samples
            }

            # Loop and copy images
            for key, samples in selected_groups.items():
                for idx, sample in enumerate(samples):

                    dataset_id = sample["dataset_id"]
                    frame_idx = sample["index"]

                    dataset_dir = os.path.join(res_dir, f'instrument_dataset_{dataset_id}')

                    # Reconstruct filename from index ordering
                    frames_dir = os.path.join(test_set_dir, f'instrument_dataset_{dataset_id}', 'left_frames')
                    filenames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png") and not f.startswith("._")])

                    if frame_idx >= len(filenames):
                        continue

                    filename = filenames[frame_idx]

                    qual_filename = 'QualRes_' + filename

                    if method == "border":
                        src_path = os.path.join(dataset_dir, 'border', 'qualitative_results', qual_filename)
                    else:
                        src_path = os.path.join(dataset_dir, 'valid', 'qualitative_results', qual_filename)

                    if not os.path.exists(src_path):
                        continue

                    new_name = f"{key}_IoU_{sample['iou']:.4f}_dataset{dataset_id}_{filename.replace('.png','')}.png"
                    dst_path = os.path.join(save_dir_method, new_name)

                    shutil.copy(src_path, dst_path)

        logger.info("Saved global qualitative samples for both methods (border & valid)")

    # Quantitative plots

    if args.skip_analysis:
        if not os.path.exists(all_ious_csv_path_border) or not os.path.exists(all_ious_csv_path_valid):
            raise FileNotFoundError(
                "IoU CSV files not found. Run without --skip-analysis first."
            )

    # Article IoUs (from provided table)
    article_ious = [0.337, 0.289, 0.483, 0.678, 0.219, 0.619, 0.325, 0.506, 0.377, 0.603]
    article_global_mean = 0.461

    # Bar comparison with three methods
    mean_ious_border = [m["border"] for m in mean_ious]
    mean_ious_valid = [m["valid_region"] for m in mean_ious]
    bar_plot_path = os.path.join(res_dir, "mean_iou_comparison.png")
    plot_bar_comparison(
        mean_ious_border,
        mean_ious_valid,
        article_ious,
        bar_plot_path
    )

    # Violin plots for both methods
    violin_plot_path_border = os.path.join(res_dir, "iou_violin_border.png")
    violin_plot_path_valid = os.path.join(res_dir, "iou_violin_valid_region.png")

    # Prepare IoU data for border and valid_region
    iou_border_dict = {d: all_ious_per_dataset[d]["border"] for d in dataset_ids}
    iou_valid_dict = {d: all_ious_per_dataset[d]["valid_region"] for d in dataset_ids}

    plot_violin_iou(iou_border_dict, dataset_ids, violin_plot_path_border)
    plot_violin_iou(iou_valid_dict, dataset_ids, violin_plot_path_valid)

    logger.info("Saved quantitative plots (bar + violin)")

if __name__ == "__main__" :
    main()