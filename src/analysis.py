import numpy as np
import os
import csv
import logging

logger = logging.getLogger(__name__)

def compute_IoU(computed_mask: np.array, GT_mask: np.array) -> float:
    logger.debug("Computing IoU")
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
