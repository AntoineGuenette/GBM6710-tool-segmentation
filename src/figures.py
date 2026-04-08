import os
import matplotlib.pyplot as plt
import numpy as np

def plot_qualitative_results(img_crop, GT_mask, prob_map, computed_mask, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0,0].imshow(img_crop)
    axes[0,0].set_title("Image originale")
    axes[0,0].axis("off")

    axes[0,1].imshow(GT_mask, cmap="gray")
    axes[0,1].set_title("Segmentation cible (GT)")
    axes[0,1].axis("off")

    axes[1,0].imshow(prob_map, cmap="gray")
    axes[1,0].set_title("Carte de probabilité")
    axes[1,0].axis("off")

    axes[1,1].imshow(computed_mask, cmap="gray")
    axes[1,1].set_title("Segmentation calculée")
    axes[1,1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_bar_comparison(mean_ious, article_ious, save_path):
    """
    Plot a bar chart comparing computed mean IoUs vs article IoUs.
    """
    labels = [f"D{i}" for i in range(1, len(mean_ious) + 1)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(x - width/2, mean_ious, width, label="IoUs calculés", color="#ACBED8")
    ax.bar(x + width/2, article_ious[:len(mean_ious)], width, label="IoUs de référence\n(Article)", color="#2E2836")

    ax.set_xlabel("Jeux de données (Datasets)")
    ax.set_ylabel("IoU moyen")
    ax.set_title("Comparaison des IoU moyens")
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
    fig, ax = plt.subplots()

    data = [all_ious[d] for d in sorted(dataset_ids)]

    parts = ax.violinplot(data, showmeans=True)

    # Set colors manually
    for pc in parts['bodies']:
        pc.set_facecolor("#ACBED8")
        pc.set_edgecolor("#ACBED8")
        pc.set_alpha(0.8)
    # Couleur des lignes (type boxplot interne)
    for key in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
        if key in parts:
            parts[key].set_color("#2E2836")

    ax.set_xticks(range(1, len(dataset_ids) + 1))
    ax.set_xticklabels([f"D{i}" for i in dataset_ids])
    ax.set_xlabel("Jeux de données (Datasets)")
    ax.set_ylabel("IoU")
    ax.set_title("Distribution des IoU par jeu de données")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
