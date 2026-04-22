import numpy as np
import os
import csv
import logging

logger = logging.getLogger(__name__)

def compute_Dice(computed_mask: np.array, GT_mask: np.array) -> float:
    """
    Compute the Dice coefficient (Dice index) between two binary masks.
    """
    TP = np.logical_and(computed_mask == 1, GT_mask == 1).sum()
    FP = np.logical_and(computed_mask == 1, GT_mask == 0).sum()
    FN = np.logical_and(computed_mask == 0, GT_mask == 1).sum()

    denom = TP + FP + FN

    if denom == 0:
        return 1.0

    iou = TP / (TP + FP + FN)
    dice = (2 * iou) / (1 + iou)
    logger.debug(f"Dice computed: {dice:.4f}")
    return dice


def compute_sensitivity(computed_mask: np.array, GT_mask: np.array) -> float:
    """
    Compute the sensitivity of the segmentation given two binary masks.
    """
    TP = np.logical_and(computed_mask == 1, GT_mask == 1).sum()
    FN = np.logical_and(computed_mask == 0, GT_mask == 1).sum()

    denom = TP + FN

    if denom == 0:
        return 1.0

    sensitivity = TP / (TP + FN)
    logger.debug(f"Sensitivity computed: {sensitivity:.4f}")
    return sensitivity


def compute_specificity(computed_mask: np.array, GT_mask: np.array) -> float:
    """
    Compute the specificity of the segmentation given two binary masks.
    """
    TN = np.logical_and(computed_mask == 0, GT_mask == 0).sum()
    FP = np.logical_and(computed_mask == 1, GT_mask == 0).sum()

    denom = TN + FP

    if denom == 0:
        return 1.0

    specificity = TN / (TN + FP)
    logger.debug(f"Specificity computed: {specificity:.4f}")
    return specificity


def compute_mean_metric(metric_list: list, metric: str) -> float:
    """
    Compute the mean metric over a list of values.
    """
    if len(metric_list) == 0:
        logger.warning(f"{metric} list is empty, returning 0.0")
        return np.nan
    
    mean_metric = float(np.mean(metric_list))
    logger.debug(f"Mean {metric} computed: {mean_metric:.4f} over {len(metric_list)} samples")
    return mean_metric


def append_dataset_result(csv_path: str, dataset_id: int, mean_dice_border: float, mean_dice_valid: float):
    """
    Append the mean Dice results of a dataset to a CSV file.
    Creates the file and header if it does not exist.
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file does not exist
        if not file_exists:
            writer.writerow(["dataset_id", "mean_Dice_border", "mean_Dice_valid_region"])

        writer.writerow([dataset_id, f"{mean_dice_border:.6f}", f"{mean_dice_valid:.6f}"])

    logger.info(f"Saved dataset {dataset_id} results to CSV: mean Dice border = {mean_dice_border:.4f}, mean Dice valid region = {mean_dice_valid:.4f}")


def append_global_mean(csv_path: str, global_mean_dice_border: float, global_mean_dice_valid: float):
    """
    Append the global mean Dice indices (mean of dataset means) to the CSV file.
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file does not exist
        if not file_exists:
            writer.writerow(["metric", "value"])

        writer.writerow(["global_mean_border", f"{global_mean_dice_border:.6f}"])
        writer.writerow(["global_mean_valid_region", f"{global_mean_dice_valid:.6f}"])

    logger.info(f"Saved global mean Dice to CSV: border = {global_mean_dice_border:.4f}, valid region = {global_mean_dice_valid:.4f}")

def save_all_dice(csv_path: str, all_dice_border: dict, all_dice_valid: dict):
    """
    Save all Dice indices per dataset into a CSV file.
    Each column corresponds to a dataset method.
    Rows are padded with empty values if datasets have different lengths.
    """
    dataset_ids = sorted(all_dice_border.keys())
    max_len = max(max(len(all_dice_border[d]), len(all_dice_valid[d])) for d in dataset_ids)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = []
        for d in dataset_ids:
            header.append(f"D{d}_border")
            header.append(f"D{d}_valid_region")
        writer.writerow(header)

        # Rows (pad with empty strings if needed)
        for i in range(max_len):
            row = []
            for d in dataset_ids:
                if i < len(all_dice_border[d]):
                    row.append(f"{all_dice_border[d][i]:.6f}")
                else:
                    row.append("")
                if i < len(all_dice_valid[d]):
                    row.append(f"{all_dice_valid[d][i]:.6f}")
                else:
                    row.append("")
            writer.writerow(row)

    logger.info(f"Saved all Dice to CSV: {csv_path}")

def load_all_dice(csv_path: str) -> (dict, dict):
    """
    Load Dice indices per dataset from a CSV file.
    Returns two dicts {dataset_id: [dice]} for border and valid_region.
    """
    all_dice_border = {}
    all_dice_valid = {}

    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader)

        dataset_ids = sorted(set(int(h.replace("D", "").replace("_border", "").replace("_valid_region", "")) for h in header))
        for d in dataset_ids:
            all_dice_border[d] = []
            all_dice_valid[d] = []

        for row in reader:
            for i, val in enumerate(row):
                if val != "":
                    col_name = header[i]
                    d = int(col_name.split("_")[0].replace("D", ""))
                    if "border" in col_name:
                        all_dice_border[d].append(float(val))
                    elif "valid_region" in col_name:
                        all_dice_valid[d].append(float(val))

    logger.info(f"Loaded Dice from CSV: {csv_path}")
    return all_dice_border, all_dice_valid
