import argparse
import os
import logging
import cv2
import numpy as np
import shutil

from segmentation import segment_tools, crop_image
from analysis import compute_Dice, compute_sensitivity, compute_specificity, compute_mean_metric, save_all_metrics, load_all_metrics
from figures import plot_qualitative_results, plot_bar_comparison, plot_violin

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
        help="Skip segmentation and Dice computation, load from CSV instead"
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
    csv_dir = os.path.join(res_dir, 'CSVs')
    figs_dir = os.path.join(res_dir, 'figs')

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    mean_metrics_csv_path = os.path.join(csv_dir, "mean_metrics.csv")

    metrics = ["Dice", "Sensitivity", "Specificity"]

    if args.skip_analysis:
        # For skip-analysis, load all metrics for each metric from their CSVs
        all_metrics_border = {}
        all_metrics_valid = {}

        for m in metrics:
            csv_path = os.path.join(csv_dir, f"all_metrics_{m.lower()}.csv")
            border_dict, valid_dict = load_all_metrics(csv_path, m)
            all_metrics_border[m] = border_dict
            all_metrics_valid[m] = valid_dict

        all_dice_per_dataset = {}
        dataset_ids = list(next(iter(all_metrics_border.values())).keys())

        for d in dataset_ids:
            metric_lists_border = {m: all_metrics_border[m][d] for m in metrics}
            metric_lists_valid = {m: all_metrics_valid[m][d] for m in metrics}

            all_dice_per_dataset[d] = {
                "border": metric_lists_border,
                "valid_region": metric_lists_valid
            }
        mean_dice = []
        for d in dataset_ids:
            mean_metrics_border = {m: compute_mean_metric(all_dice_per_dataset[d]["border"][m], m) for m in metrics}
            mean_metrics_valid = {m: compute_mean_metric(all_dice_per_dataset[d]["valid_region"][m], m) for m in metrics}
            mean_dice.append({
                "border": mean_metrics_border,
                "valid_region": mean_metrics_valid
            })
    else:
        # Store global Dice results for best/worst/median selection
        all_results = []

        all_dice_per_dataset = {}
        mean_dice = []

        # Iterate over all datasets (1 to 10)
        for i in range(1, 11):

            logger.info(f"-> PROCESSING DATASET {i} <-")
            metric_lists_border = {m: [] for m in metrics}
            metric_lists_valid = {m: [] for m in metrics}

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

                    dice_valid = compute_Dice(pred_mask_valid, gt_mask)
                    sens_valid = compute_sensitivity(pred_mask_valid, gt_mask)
                    spec_valid = compute_specificity(pred_mask_valid, gt_mask)

                    metric_lists_valid["Dice"].append(dice_valid)
                    metric_lists_valid["Sensitivity"].append(sens_valid)
                    metric_lists_valid["Specificity"].append(spec_valid)

                    # Load BORDER mask
                    pred_mask_border = cv2.imread(mask_path_border, cv2.IMREAD_GRAYSCALE)
                    pred_mask_border = (pred_mask_border > 0).astype(np.uint8)

                    dice_border = compute_Dice(pred_mask_border, gt_mask)
                    sens_border = compute_sensitivity(pred_mask_border, gt_mask)
                    spec_border = compute_specificity(pred_mask_border, gt_mask)

                    metric_lists_border["Dice"].append(dice_border)
                    metric_lists_border["Sensitivity"].append(sens_border)
                    metric_lists_border["Specificity"].append(spec_border)

                    prob_map_path = os.path.join(dataset_dir, 'prob_maps', 'prob_' + filename)

                    # Store for global qualitative selection
                    all_results.append({
                        "dice_valid": dice_valid,
                        "dice_border": dice_border,
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
                        logger.warning(f"Missing mask for {filename}, skipping Dice computation")
                        computed_mask_bin_border = None
                        computed_mask_bin_valid = None
                        GT_mask_bin = None

                    if computed_mask_bin_border is not None and GT_mask_bin is not None:
                        # Compute Dice, Sensitivity, Specificity for border
                        dice_border = compute_Dice(computed_mask_bin_border, GT_mask_bin)
                        sens_border = compute_sensitivity(computed_mask_bin_border, GT_mask_bin)
                        spec_border = compute_specificity(computed_mask_bin_border, GT_mask_bin)

                        metric_lists_border["Dice"].append(dice_border)
                        metric_lists_border["Sensitivity"].append(sens_border)
                        metric_lists_border["Specificity"].append(spec_border)
                    else:
                        dice_border = None

                    if computed_mask_bin_valid is not None and GT_mask_bin is not None:
                        # Compute Dice, Sensitivity, Specificity for valid region
                        dice_valid = compute_Dice(computed_mask_bin_valid, GT_mask_bin)
                        sens_valid = compute_sensitivity(computed_mask_bin_valid, GT_mask_bin)
                        spec_valid = compute_specificity(computed_mask_bin_valid, GT_mask_bin)

                        metric_lists_valid["Dice"].append(dice_valid)
                        metric_lists_valid["Sensitivity"].append(sens_valid)
                        metric_lists_valid["Specificity"].append(spec_valid)
                    else:
                        dice_valid = None

                    prob_map_path = os.path.join(dataset_dir, 'prob_maps', 'prob_' + filename)

                    all_results.append({
                        "dice_valid": dice_valid,
                        "dice_border": dice_border,
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
                        title=f"{filename}\nDice valid: {dice_valid:.4f}",
                        save_path=qual_res_file_path_valid
                    )

                    # Qualitative result for BORDER method
                    plot_qualitative_results(
                        img_crop_border,
                        GT_mask_bin,
                        prob_map_border,
                        computed_mask_bin_border,
                        title=f"{filename}\nDice border: {dice_border:.4f}",
                        save_path=qual_res_file_path_border
                    )

            # Compute mean for all metrics for both methods
            mean_metrics_border = {m: compute_mean_metric(metric_lists_border[m], m) for m in metrics}
            mean_metrics_valid = {m: compute_mean_metric(metric_lists_valid[m], m) for m in metrics}
            all_dice_per_dataset[i] = {
                "border": metric_lists_border,
                "valid_region": metric_lists_valid
            }
            mean_dice.append({
                "border": mean_metrics_border,
                "valid_region": mean_metrics_valid
            })
            logger.info(f"Dataset {i} - Means (border): {mean_metrics_border} | (valid): {mean_metrics_valid}")

            # Save mean metrics in a CSV file with all metrics
            file_exists = os.path.isfile(mean_metrics_csv_path)
            with open(mean_metrics_csv_path, 'a', newline='') as f:
                import csv
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        "dataset_id",
                        "Dice_border", "Dice_valid",
                        "Sensitivity_border", "Sensitivity_valid",
                        "Specificity_border", "Specificity_valid"
                    ])

                writer.writerow([
                    i,
                    f"{mean_metrics_border['Dice']:.6f}", f"{mean_metrics_valid['Dice']:.6f}",
                    f"{mean_metrics_border['Sensitivity']:.6f}", f"{mean_metrics_valid['Sensitivity']:.6f}",
                    f"{mean_metrics_border['Specificity']:.6f}", f"{mean_metrics_valid['Specificity']:.6f}"
                ])

        # Save all metrics to CSV
        metrics_border = {m: {} for m in metrics}
        metrics_valid = {m: {} for m in metrics}

        for dataset_id, metric_dict in all_dice_per_dataset.items():
            for m in metrics:
                metrics_border[m][dataset_id] = metric_dict["border"][m]
                metrics_valid[m][dataset_id] = metric_dict["valid_region"][m]

        # Save each metric into its own CSV file
        for m in metrics:
            csv_path = os.path.join(csv_dir, f"all_metrics_{m.lower()}.csv")
            save_all_metrics(csv_path, metrics_border[m], metrics_valid[m], m)

    dataset_ids = list(all_dice_per_dataset.keys())

    # Compute global mean of means for both methods (Dice only, for CSV and logs)
    global_mean_dice_border = float(np.mean([m["border"]["Dice"] for m in mean_dice])) if len(mean_dice) > 0 else 0.0
    global_mean_dice_valid = float(np.mean([m["valid_region"]["Dice"] for m in mean_dice])) if len(mean_dice) > 0 else 0.0
    global_mean_sens_border = float(np.mean([m["border"]["Sensitivity"] for m in mean_dice])) if len(mean_dice) > 0 else 0.0
    global_mean_sens_valid = float(np.mean([m["valid_region"]["Sensitivity"] for m in mean_dice])) if len(mean_dice) > 0 else 0.0
    global_mean_spec_border = float(np.mean([m["border"]["Specificity"] for m in mean_dice])) if len(mean_dice) > 0 else 0.0
    global_mean_spec_valid = float(np.mean([m["valid_region"]["Specificity"] for m in mean_dice])) if len(mean_dice) > 0 else 0.0
    logger.info(f"Global mean Dice Border (mean of means): {global_mean_dice_border:.4f}")
    logger.info(f"Global mean Dice Valid Region (mean of means): {global_mean_dice_valid:.4f}")
    logger.info(f"Global mean Sensitivity Border (mean of means): {global_mean_sens_border:.4f}")
    logger.info(f"Global mean Sensitivity Valid Region (mean of means): {global_mean_sens_valid:.4f}")
    logger.info(f"Global mean Specificity Border (mean of means): {global_mean_spec_border:.4f}")
    logger.info(f"Global mean Specificity Valid Region (mean of means): {global_mean_spec_valid:.4f}")

    # Save to CSV global means (all metrics, only Dice filled for global_mean row)
    with open(mean_metrics_csv_path, 'a', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow([
            "global_mean",
            f"{global_mean_dice_border:.6f}", f"{global_mean_dice_valid:.6f}",
            f"{global_mean_sens_border:.6f}", f"{global_mean_sens_valid:.6f}",
            f"{global_mean_spec_border:.6f}", f"{global_mean_spec_valid:.6f}",
        ])

    # Global qualitative selection for BOTH methods (border + valid)
    if len(all_dice_per_dataset) > 0:

        global_vis_dir = os.path.join(res_dir, "global_qualitative")
        border_dir = os.path.join(global_vis_dir, "border")
        valid_dir = os.path.join(global_vis_dir, "valid")
        os.makedirs(border_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

        for method in ["border", "valid_region"]:

            # Select correct directory and Dice key
            if method == "border":
                save_dir_method = border_dir
                dice_key = "border"
            else:
                save_dir_method = valid_dir
                dice_key = "valid_region"

            # Build flat list of results with Dice and metadata, casting to float and skipping invalid values
            method_results = []
            for d in all_dice_per_dataset:
                for idx, dice in enumerate(all_dice_per_dataset[d][dice_key]):
                    if dice is None:
                        continue
                    try:
                        dice_val = float(dice)
                    except (ValueError, TypeError):
                        continue
                    method_results.append({
                        "dice": dice_val,
                        "dataset_id": d,
                        "index": idx
                    })

            if len(method_results) == 0:
                continue

            # Sort by Dice
            method_results_sorted = sorted(method_results, key=lambda x: x["dice"])
            n = len(method_results_sorted)

            # Exclude Dice == 0 for minimum samples
            non_zero_results = [r for r in method_results_sorted if r["dice"] > 0]

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

                    new_name = f"{key}_Dice_{sample['dice']:.4f}_dataset{dataset_id}_{filename.replace('.png','')}.png"
                    dst_path = os.path.join(save_dir_method, new_name)

                    shutil.copy(src_path, dst_path)

        logger.info("Saved global qualitative samples for both methods (border & valid)")

    # Quantitative plots

    # Article IoU (from provided table)
    article_ious = [0.337, 0.289, 0.483, 0.678, 0.219, 0.619, 0.325, 0.506, 0.377, 0.603]
    article_mean_iou = 0.461

    # Convert IoU (article) to Dice:
    article_dice = [(2 * iou) / (1 + iou) for iou in article_ious]
    article_mean_dice = (2 * article_mean_iou) / (1 + article_mean_iou)

    # Bar comparison and violin plots for all metrics
    for metric in metrics:
        mean_metric_border = [m["border"][metric] for m in mean_dice]
        mean_metric_valid = [m["valid_region"][metric] for m in mean_dice]

        plot_bar_comparison(
            mean_metric_border + [np.mean(mean_metric_border)],
            mean_metric_valid + [np.mean(mean_metric_valid)],
            article_dice[:len(mean_dice)] + [article_mean_dice] if metric == "Dice" else [0]*(len(mean_metric_border)+1),
            metric,
            os.path.join(figs_dir, f"{metric.lower()}_bar.png")
        )

    for metric in metrics:
        metric_border_dict = {d: all_dice_per_dataset[d]["border"][metric] for d in dataset_ids}
        metric_valid_dict = {d: all_dice_per_dataset[d]["valid_region"][metric] for d in dataset_ids}

        plot_violin(metric_border_dict, metric, dataset_ids, os.path.join(figs_dir, f"{metric.lower()}_violin_border.png"))
        plot_violin(metric_valid_dict, metric, dataset_ids, os.path.join(figs_dir, f"{metric.lower()}_violin_valid.png"))

    logger.info("Saved quantitative plots (bar + violin)")

if __name__ == "__main__" :
    main()