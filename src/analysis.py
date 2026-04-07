import numpy as np

def compute_IoU(computed_mask: np.array, GT_mask: np.array) -> float:
    TP = np.logical_and(computed_mask == 1, GT_mask == 1).sum()
    FP = np.logical_and(computed_mask == 1, GT_mask == 0).sum()
    FN = np.logical_and(computed_mask == 0, GT_mask == 1).sum()

    denom = TP + FP + FN

    # Fix IoU=1 if both masks are all empty
    if denom == 0:
        return 1.0

    return TP / (TP + FP + FN)
