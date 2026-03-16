import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

def crop_image(img:np.array, crop_pix_x:int, crop_pix_y:int, width:int, height:int) -> np.array:
    return img[crop_pix_y:crop_pix_y+height, crop_pix_x:crop_pix_x+width]

def color_filtering(img_rgb:np.array) -> np.array:
    # Convert RGB image to HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    h = img_hsv[:,:,0] # possible values : 0-179
    s = img_hsv[:,:,1] # possible values : 0-255
    v = img_hsv[:,:,2] # possible values : 0-255

    # Define Opponent Color space
    R = img_rgb[:,:,0].astype(np.float32) # possible values : 0-255
    G = img_rgb[:,:,1].astype(np.float32) # possible values : 0-255
    B = img_rgb[:,:,2].astype(np.float32) # possible values : 0-255
    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2*B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)

    # Define thresholds to distinguish metallic tools
    # In HSV
    mask_hue = h > 70 # eliminates red/oranges
    mask_sat = s < 60 # low saturation
    mask_val = v > 40 # eliminates dark spots
    # In Opponent Color space
    mask_o1 = np.abs(O1) < 15
    mask_o2 = np.abs(O2) < 15

    # Combine masks
    mask =  mask_hue & mask_sat & mask_val & mask_o1 & mask_o2
    mask = mask.astype(np.uint8) * 255

    # Morphologic filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def run_grabcut(img_rgb: np.array, init_mask: np.array, iterations: int = 5) -> np.array:
    # Prepare GrabCut mask
    grabcut_mask = np.where(init_mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')

    # Initialize background and foreground models required by GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut using the mask initialization
    cv2.grabCut(img_rgb, grabcut_mask, None, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)

    # Convert GrabCut output to binary mask
    grabcut_result = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), # we consider both background (BG) and probable background (PR_BG)
        255,
        0
    ).astype('uint8')

    return grabcut_result

def extract_edges(img_rgb):
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Canny filter
    edge_mask = cv2.Canny(img_gray, 20, 200)

    return edge_mask

def extract_elongated_shapes(mask):
    # Label all separate regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Initialization
    shape_mask = np.zeros_like(mask, dtype=np.float32)

    for i in range(1, num_labels):
        # Extract width/height of each region
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Only elongated regions are kept
        elongation = max(width, height) / (min(width, height) + 1e-5)
        if elongation > 1.5:  
            shape_mask[labels == i] = 255.0

    return shape_mask

def extract_border_regions(mask):
    # Label all separate regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Initialization
    h, w = mask.shape
    border_feature = np.zeros_like(mask, dtype=np.float32)

    for i in range(1, num_labels):
        # Extract stats of each region
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Only the regions touching the borfer are kept
        touches_border = (
            x == 0 or
            y == 0 or
            (x + width) >= w-1 or
            (y + height) >= h-1
        )
        if touches_border:
            border_feature[labels == i] = 255.0

    return border_feature

def compute_blur_score(img_rgb):
    # Convert to gray scale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Laplacian filter
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)

    # The blur score is the variance of the Laplacian response
    blur_score = laplacian.var()
    return blur_score

def blur_factor(blur_score, min_blur=50, max_blur=150):
    """
    Normalize blur score to a factor between 0 and 1.
    0 -> very blurry image
    1 -> sharp image
    """
    factor = (blur_score - min_blur) / (max_blur - min_blur)
    factor = np.clip(factor, 0.0, 1.0)
    return factor


def compute_dynamic_weights(bf):
    w_color = 0.25
    w_grabcut = 0.3
    w_shape = 0.3
    w_edge = 0.15 * bf

    total = w_color + w_grabcut + w_shape + w_edge

    return (
        w_color/total,
        w_grabcut/total,
        w_edge/total,
        w_shape/total
    )


if __name__ == '__main__' :
    # Define paths
    script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(script_path)
    repo_dir = os.path.join(src_dir, '..')
    data_dir = os.path.join(repo_dir, 'data')

    # Select image to analyse
    # img_path = os.path.join(data_dir, 'dataset_1', 'frame239.png')
    img_path = os.path.join(data_dir, 'dataset_2', 'frame260.png')
    # img_path = os.path.join(data_dir, 'dataset_3', 'frame245.png')
    # img_path = os.path.join(data_dir, 'dataset_4', 'frame238.png')
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop image
    img_crop = crop_image(img_rgb, 328, 37, 1264, 1010)

    # Define possible tool areas
    color_mask = color_filtering(img_crop)

    # Refine segmentation with GrabCut
    grabcut_mask = run_grabcut(img_crop, color_mask)

    # Extract features
    edge_mask = extract_edges(img_crop)
    shape_mask = extract_elongated_shapes(color_mask)
    border_mask = extract_border_regions(color_mask)

    # Convert masks to probabilities
    color_prob = color_mask.astype(np.float32) / 255
    grabcut_prob = grabcut_mask.astype(np.float32) / 255
    edge_prob = edge_mask.astype(np.float32) / 255
    shape_prob = shape_mask.astype(np.float32) / 255
    border_prob = border_mask.astype(np.float32) / 255

    # Compute blur level and dynamic weights
    blur_score = compute_blur_score(img_crop)
    bf = blur_factor(blur_score)
    w_color, w_grabcut, w_edge, w_shape = compute_dynamic_weights(bf)

    # Compute score
    prob_map = (
        w_color * color_prob +
        w_grabcut * grabcut_prob +
        w_edge * edge_prob +
        w_shape * shape_prob
    )
    prob_map *= border_prob

    # Threshold
    final_mask = (prob_map > 0.5).astype(np.uint8) * 255
    
    # Show images and computed features
    fig, axes = plt.subplots(2, 4, figsize=(14,8))

    axes[0,0].imshow(img_crop)
    axes[0,0].set_title("Image originale")
    axes[0,0].axis("off")

    axes[0,1].imshow(color_mask, cmap="gray")
    axes[0,1].set_title("Color mask")
    axes[0,1].axis("off")

    axes[0,2].imshow(grabcut_mask, cmap="gray")
    axes[0,2].set_title("GrabCut mask")
    axes[0,2].axis("off")

    axes[0,3].imshow(prob_map, cmap="gray")
    axes[0,3].set_title("Probability map")
    axes[0,3].axis("off")

    axes[1,0].imshow(edge_mask, cmap="gray")
    axes[1,0].set_title("Edge feature")
    axes[1,0].axis("off")

    axes[1,1].imshow(shape_mask, cmap="gray")
    axes[1,1].set_title("Shape feature")
    axes[1,1].axis("off")

    axes[1,2].imshow(border_mask, cmap="gray")
    axes[1,2].set_title("Border feature")
    axes[1,2].axis("off")

    axes[1,3].imshow(final_mask, cmap="gray")
    axes[1,3].set_title("Final mask")
    axes[1,3].axis("off")

    plt.suptitle(f"Image processing\nBlur score: {blur_score:.2f} | Blur factor: {bf:.2f}")
    plt.tight_layout()
    plt.show()
