import numpy as np
import os
import csv
import logging

logger = logging.getLogger(__name__)

def compute_IoU(computed_mask: np.array, GT_mask: np.array) -> float:
    TP = np.logical_and(computed_mask == 1, GT_mask == 1).sum()
    FP = np.logical_and(computed_mask == 1, GT_mask == 0).sum()
    FN = np.logical_and(computed_mask == 0, GT_mask == 1).sum()

    denom = TP + FP + FN

    # Fix IoU=1 if both masks are all empty
    if denom == 0:
        return 1.0

    iou = TP / (TP + FP + FN)
    logger.debug(f"IoU computed: TP={TP}, FP={FP}, FN={FN}, IoU={iou:.4f}")
    return iou


def compute_mean_IoU(iou_list: list) -> float:
    """
    Compute the mean IoU over a list of IoU values.
    Returns 0.0 if the list is empty.
    """
    if len(iou_list) == 0:
        logger.warning("IoU list is empty, returning 0.0")
        return 0.0
    
    mean_iou = float(np.mean(iou_list))
    logger.debug(f"Mean IoU computed: {mean_iou:.4f} over {len(iou_list)} samples")
    return mean_iou


def append_dataset_result(csv_path: str, dataset_id: int, mean_iou_border: float, mean_iou_valid: float):
    """
    Append the mean IoU results of a dataset to a CSV file.
    Creates the file and header if it does not exist.
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file does not exist
        if not file_exists:
            writer.writerow(["dataset_id", "mean_IoU_border", "mean_IoU_valid_region"])

        writer.writerow([dataset_id, f"{mean_iou_border:.6f}", f"{mean_iou_valid:.6f}"])

    logger.info(f"Saved dataset {dataset_id} results to CSV: mean IoU border = {mean_iou_border:.4f}, mean IoU valid region = {mean_iou_valid:.4f}")

def append_global_mean(csv_path: str, global_mean_iou_border: float, global_mean_iou_valid: float):
    """
    Append the global mean IoUs (mean of dataset means) to the CSV file.
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file does not exist
        if not file_exists:
            writer.writerow(["metric", "value"])

        writer.writerow(["global_mean_border", f"{global_mean_iou_border:.6f}"])
        writer.writerow(["global_mean_valid_region", f"{global_mean_iou_valid:.6f}"])

    logger.info(f"Saved global mean IoUs to CSV: border = {global_mean_iou_border:.4f}, valid region = {global_mean_iou_valid:.4f}")

def save_all_ious(csv_path: str, all_ious_border: dict, all_ious_valid: dict):
    """
    Save all IoUs per dataset into a CSV file.
    Each column corresponds to a dataset method.
    Rows are padded with empty values if datasets have different lengths.
    """
    dataset_ids = sorted(all_ious_border.keys())
    max_len = max(max(len(all_ious_border[d]), len(all_ious_valid[d])) for d in dataset_ids)

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
                if i < len(all_ious_border[d]):
                    row.append(f"{all_ious_border[d][i]:.6f}")
                else:
                    row.append("")
                if i < len(all_ious_valid[d]):
                    row.append(f"{all_ious_valid[d][i]:.6f}")
                else:
                    row.append("")
            writer.writerow(row)

    logger.info(f"Saved all IoUs to CSV: {csv_path}")

def load_all_ious(csv_path: str) -> (dict, dict):
    """
    Load IoUs per dataset from a CSV file.
    Returns two dicts {dataset_id: [ious]} for border and valid_region.
    """
    all_ious_border = {}
    all_ious_valid = {}

    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader)

        dataset_ids = sorted(set(int(h.replace("D", "").replace("_border", "").replace("_valid_region", "")) for h in header))
        for d in dataset_ids:
            all_ious_border[d] = []
            all_ious_valid[d] = []

        for row in reader:
            for i, val in enumerate(row):
                if val != "":
                    col_name = header[i]
                    d = int(col_name.split("_")[0].replace("D", ""))
                    if "border" in col_name:
                        all_ious_border[d].append(float(val))
                    elif "valid_region" in col_name:
                        all_ious_valid[d].append(float(val))

    logger.info(f"Loaded IoUs from CSV: {csv_path}")
    return all_ious_border, all_ious_valid
