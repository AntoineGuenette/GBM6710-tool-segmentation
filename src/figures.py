import os
import matplotlib.pyplot as plt
import numpy as np

def plot_qualitative_results(img_crop, GT_mask, prob_map, computed_mask, title, save_path):
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


def plot_bar_comparison(mean_ious_border, mean_ious_valid, article_ious, save_path):
    """
    Plot a bar chart comparing IoUs per dataset + global for 3 methods
    """

    # Nombre de datasets
    n = len(mean_ious_border)

    # Labels avec Global
    labels = [f"D{i+1}" for i in range(n)]
    labels.append("Global")

    # Ajouter la moyenne globale à la fin
    mean_ious_border_ext = mean_ious_border + [np.mean(mean_ious_border)]
    mean_ious_valid_ext = mean_ious_valid + [np.mean(mean_ious_valid)]
    article_ious_ext = article_ious[:n] + [np.mean(article_ious[:n])]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, article_ious_ext, width, label="Référence (Article)", color="#2E2836")
    bars2 = ax.bar(x, mean_ious_border_ext, width, label="Ré-implémentation", color="#6A8CAF")
    bars3 = ax.bar(x + width, mean_ious_valid_ext, width, label="Méthode améliorée", color="#ACBED8")
    
    # Labels au-dessus des barres
    for bars in [bars1, bars2, bars3]:
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
    ax.set_ylabel("IoU moyen")
    ax.set_title(r"$\bf{Figure\ 10}$ – Comparaison des IoUs moyens")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_violin_iou(all_ious, dataset_ids, save_path):
    """
    Plot violin plot of IoU distributions per dataset.
    """
    fig, ax = plt.subplots(figsize=(12,6))

    data = [all_ious[d] for d in sorted(dataset_ids)]

    # Add global distribution (all IoUs pooled)
    global_data = [iou for d in dataset_ids for iou in all_ious[d]]
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
    ax.set_ylabel("IoU")
    ax.set_title(r"$\bf{Figure\ 9}$ – Distribution des IoUs par jeu de données")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
