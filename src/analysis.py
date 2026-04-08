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


def append_dataset_result(csv_path: str, dataset_id: int, mean_iou: float):
    """
    Append the mean IoU result of a dataset to a CSV file.
    Creates the file and header if it does not exist.
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file does not exist
        if not file_exists:
            writer.writerow(["dataset_id", "mean_IoU"])

        writer.writerow([dataset_id, f"{mean_iou:.6f}"])

    logger.info(f"Saved dataset {dataset_id} result to CSV: mean IoU = {mean_iou:.4f}")

def append_global_mean(csv_path: str, global_mean_iou: float):
    """
    Append the global mean IoU (mean of dataset means) to the CSV file.
    """
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["global_mean", f"{global_mean_iou:.6f}"])

    logger.info(f"Saved global mean IoU to CSV: {global_mean_iou:.4f}")

def save_all_ious(csv_path: str, all_ious: dict):
    """
    Save all IoUs per dataset into a CSV file.
    Each column corresponds to a dataset.
    Rows are padded with empty values if datasets have different lengths.
    """
    dataset_ids = sorted(all_ious.keys())
    max_len = max(len(all_ious[d]) for d in dataset_ids)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = [f"D{d}" for d in dataset_ids]
        writer.writerow(header)

        # Rows (pad with empty strings if needed)
        for i in range(max_len):
            row = []
            for d in dataset_ids:
                if i < len(all_ious[d]):
                    row.append(f"{all_ious[d][i]:.6f}")
                else:
                    row.append("")
            writer.writerow(row)

    logger.info(f"Saved all IoUs to CSV: {csv_path}")

def load_all_ious(csv_path: str) -> dict:
    """
    Load IoUs per dataset from a CSV file.
    Returns a dict {dataset_id: [ious]}.
    """
    all_ious = {}

    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader)

        dataset_ids = [int(h.replace("D", "")) for h in header]
        for d in dataset_ids:
            all_ious[d] = []

        for row in reader:
            for i, val in enumerate(row):
                if val != "":
                    all_ious[dataset_ids[i]].append(float(val))

    logger.info(f"Loaded IoUs from CSV: {csv_path}")
    return all_ious
