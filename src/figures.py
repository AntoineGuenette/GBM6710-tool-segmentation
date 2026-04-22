import os
import matplotlib.pyplot as plt
import numpy as np

from segmentation import crop_image

def plot_qualitative_results(
    img_crop: np.ndarray,
    GT_mask: np.ndarray,
    prob_map: np.ndarray,
    computed_mask: np.ndarray,
    title: str,
    save_path: str
) -> None:
    """
    Plot qualitative segmentation results (image, GT, probability map, prediction) in a 2x2 grid and save to disk.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # --- Safety checks for img_crop ---
    if img_crop is None:
        raise ValueError("img_crop is None (image not loaded or skipped improperly)")

    if not isinstance(img_crop, np.ndarray):
        raise TypeError(f"img_crop is not a numpy array: {type(img_crop)}")

    if img_crop.dtype == object:
        raise TypeError("img_crop has dtype=object (invalid image structure)")

    if img_crop.ndim not in [2, 3]:
        raise ValueError(f"img_crop has invalid number of dimensions: {img_crop.ndim}")

    axes[0,0].imshow(img_crop)
    axes[0,0].set_title("Image originale")
    axes[0,0].axis("off")

    if GT_mask is None:
        raise ValueError("GT_mask is None")
    axes[0,1].imshow(GT_mask, cmap="gray")
    axes[0,1].set_title("Segmentation cible (GT)")
    axes[0,1].axis("off")

    if prob_map is None:
        raise ValueError("prob_map is None")
    axes[1,0].imshow(prob_map, cmap="gray")
    axes[1,0].set_title("Carte de probabilité")
    axes[1,0].axis("off")

    if computed_mask is None:
        raise ValueError("computed_mask is None")
    axes[1,1].imshow(computed_mask, cmap="gray")
    axes[1,1].set_title("Segmentation calculée")
    axes[1,1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_bar_comparison(
    mean_metric_border: list[float],
    mean_metric_valid: list[float],
    article_metric: list[float],
    metric_name: str,
    save_path: str
) -> None:
    """
    Plot bar comparison of mean metric values for border and valid methods (and optionally article reference).
    """

    n_with_global = len(mean_metric_border)
    n = n_with_global - 1

    labels = [f"D{i+1}" for i in range(n)]
    labels.append("Global")

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    has_article = any(val != 0 for val in article_metric)

    if has_article:
        bars1 = ax.bar(x - width, article_metric[:n_with_global], width,
                       label="Référence (Article)", color="#2E2836")
        bars2 = ax.bar(x, mean_metric_border, width,
                       label="Ré-implémentation", color="#6A8CAF")
        bars3 = ax.bar(x + width, mean_metric_valid, width,
                       label="Méthode améliorée", color="#ACBED8")
        bar_groups = [bars1, bars2, bars3]
    else:
        bars2 = ax.bar(x - width/2, mean_metric_border, width,
                       label="Ré-implémentation", color="#6A8CAF")
        bars3 = ax.bar(x + width/2, mean_metric_valid, width,
                       label="Méthode améliorée", color="#ACBED8")
        bar_groups = [bars2, bars3]

    for bars in bar_groups:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=8
            )

    ax.set_xlabel("Jeux de données (Datasets)")
    ax.set_ylabel(f"{metric_name} moyen")
    ax.set_title(f"Comparaison du {metric_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_violin(
    metric_list: dict[int, list[float]],
    metric: str,
    dataset_ids: list[int],
    save_path: str
) -> None:
    """
    Plot violin and boxplot distributions of a metric per dataset and globally.
    """
    fig, ax = plt.subplots(figsize=(12,6))

    data = [metric_list[d] for d in sorted(dataset_ids)]

    # Add global distribution
    global_data = [met for d in dataset_ids for met in metric_list[d]]
    data.append(global_data)

    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    # Set colors manually
    for pc in parts['bodies']:
        pc.set_facecolor("#ACBED8")
        pc.set_edgecolor("none")
        pc.set_alpha(0.8)

    # Remove default violin lines to avoid overlap with boxplot
    for key in ['cbars', 'cmins', 'cmaxes']:
        if key in parts:
            parts[key].set_visible(False)

    # Overlay boxplot
    box = ax.boxplot(
        data,
        positions=range(1, len(data) + 1),
        widths=0.15,
        patch_artist=True,
        showfliers=False
    )

    # Style boxplot
    for patch in box['boxes']:
        patch.set_facecolor("none")
        patch.set_edgecolor("#2E2836")
        patch.set_linewidth(1.5)

    for element in ['whiskers', 'caps', 'medians']:
        for line in box[element]:
            line.set_color("#2E2836")
            line.set_linewidth(1.5)

    ax.set_xticks(range(1, len(dataset_ids) + 2))
    labels = [f"D{i}" for i in dataset_ids]
    labels.append("Global")
    ax.set_xticklabels(labels)
    ax.set_xlabel("Jeux de données (Datasets)")
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Distribution des {metric} par jeu de données")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_global_qualitative_grid(
    samples: dict[str, list[dict]],
    method: str,
    save_path: str,
    gt_dir: str,
    test_dir: str,
    res_dir: str,
    metric_name: str
) -> None:
    """
    Create a 3x3 qualitative grid (min, median, max) showing image, GT and prediction for a given method and metric.
    """
    import cv2

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    titles = ["Minimum", "Médiane", "Maximum"]

    for col, (key, sample_list) in enumerate(samples.items()):
        if len(sample_list) == 0:
            continue

        sample = sample_list[0]  # take representative

        dataset_id = sample["dataset_id"]
        frame_idx = sample["index"]

        frames_dir = os.path.join(test_dir, f'instrument_dataset_{dataset_id}', 'left_frames')
        filenames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png") and not f.startswith("._")])

        if frame_idx >= len(filenames):
            continue

        filename = filenames[frame_idx]

        # Load image
        frame_path = os.path.join(frames_dir, filename)
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image(img, 328, 37, 1264, 1010)

        # Load GT
        gt_path = os.path.join(gt_dir, f'instrument_dataset_{dataset_id}', 'BinarySegmentation', filename)
        gt = cv2.imread(gt_path)
        gt = crop_image(gt, 328, 37, 1264, 1010)

        # Load prediction
        if method == "border":
            pred_path = os.path.join(res_dir, f'instrument_dataset_{dataset_id}', 'border', 'binary_segmentations', 'bin_' + filename)
        else:
            pred_path = os.path.join(res_dir, f'instrument_dataset_{dataset_id}', 'valid', 'binary_segmentations', 'bin_' + filename)

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        axes[0, col].imshow(img)
        axes[1, col].imshow(gt, cmap="gray")
        axes[2, col].imshow(pred, cmap="gray")

        metric_val = sample.get("metric", None)
        if metric_val is not None:
            title_str = f"{titles[col]}\nD{dataset_id} - Frame {filename}\n{metric_name}: {metric_val:.4f}"
        else:
            title_str = f"{titles[col]}\nD{dataset_id} - Frame {filename}"

        axes[0, col].set_title(title_str)

        for row in range(3):
            axes[row, col].axis("off")

    row_labels = ["Image", "GT", "Segmentation"]
    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    