import numpy as np
import csv
import logging

logger = logging.getLogger(__name__)

def compute_Dice(computed_mask: np.ndarray, GT_mask: np.ndarray) -> float:
    """
    Compute the Dice coefficient between two binary masks.
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

def compute_sensitivity(computed_mask: np.ndarray, GT_mask: np.ndarray) -> float:
    """
    Compute the sensitivity (recall) between two binary masks.
    """
    TP = np.logical_and(computed_mask == 1, GT_mask == 1).sum()
    FN = np.logical_and(computed_mask == 0, GT_mask == 1).sum()

    denom = TP + FN

    if denom == 0:
        return 1.0

    sensitivity = TP / (TP + FN)
    logger.debug(f"Sensitivity computed: {sensitivity:.4f}")
    return sensitivity

def compute_specificity(computed_mask: np.ndarray, GT_mask: np.ndarray) -> float:
    """
    Compute the specificity between two binary masks.
    """
    TN = np.logical_and(computed_mask == 0, GT_mask == 0).sum()
    FP = np.logical_and(computed_mask == 1, GT_mask == 0).sum()

    denom = TN + FP

    if denom == 0:
        return 1.0

    specificity = TN / (TN + FP)
    logger.debug(f"Specificity computed: {specificity:.4f}")
    return specificity

def compute_mean_metric(metric_list: list[float], metric_name: str) -> float:
    """
    Compute the mean value of a metric list.
    """
    if len(metric_list) == 0:
        logger.warning(f"{metric_name} list is empty, returning 0.0")
        return np.nan

    mean_metric = float(np.mean(metric_list))
    logger.debug(f"Mean {metric_name} computed: {mean_metric:.4f} over {len(metric_list)} samples")
    return mean_metric

def save_all_metrics(
    csv_path: str,
    all_metric_border: dict[int, list[float]],
    all_metric_valid: dict[int, list[float]],
    metric_name: str
) -> None:
    """
    Save all metric values per dataset into a CSV file.
    """
    dataset_ids = sorted(all_metric_border.keys())
    max_len = max(max(len(all_metric_border[d]), len(all_metric_valid[d])) for d in dataset_ids)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Write header row
        header = []
        for d in dataset_ids:
            header.append(f"D{d}_{metric_name}_border")
            header.append(f"D{d}_{metric_name}_valid_region")
        writer.writerow(header)

        # Write padded data rows
        for i in range(max_len):
            row = []
            for d in dataset_ids:
                if i < len(all_metric_border[d]):
                    row.append(f"{all_metric_border[d][i]:.6f}")
                else:
                    row.append("")
                if i < len(all_metric_valid[d]):
                    row.append(f"{all_metric_valid[d][i]:.6f}")
                else:
                    row.append("")
            writer.writerow(row)

    logger.info(f"Saved all {metric_name} to CSV: {csv_path}")

def load_all_metrics(
    csv_path: str,
    metric_name: str
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """
    Load metric values per dataset from a CSV file.
    """
    all_metric_border = {}
    all_metric_valid = {}

    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Extract dataset identifiers
        dataset_ids = set()
        for h in header:
            if not h.startswith("D"):
                continue
            parts = h.split("_")
            try:
                d_id = int(parts[0].replace("D", ""))
                dataset_ids.add(d_id)
            except (ValueError, IndexError):
                continue
        dataset_ids = sorted(dataset_ids)

        # Initialize metric containers
        for d in dataset_ids:
            all_metric_border[d] = []
            all_metric_valid[d] = []

        # Read metric values
        for row in reader:
            for i, val in enumerate(row):
                if val != "":
                    col_name = header[i]
                    d = int(col_name.split("_")[0].replace("D", ""))
                    if f"{metric_name}_border" in col_name:
                        all_metric_border[d].append(float(val))
                    elif f"{metric_name}_valid_region" in col_name:
                        all_metric_valid[d].append(float(val))

    logger.info(f"Loaded {metric_name} from CSV: {csv_path}")
    return all_metric_border, all_metric_valid
