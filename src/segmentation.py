import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def segment_tools(img_path: str, dataset_dir: str=None, save_subdir: str=None, debug: bool=False, method: str="valid_region"):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop image
    img_crop = crop_image(img_rgb, 328, 37, 1264, 1010)
    if img_crop is None or not isinstance(img_crop, np.ndarray):
        raise ValueError("Cropping failed: img_crop is invalid")
    if img_crop.dtype == object:
        raise TypeError("img_crop has dtype=object after cropping")

    # Define possible tool areas
    color_mask = color_filtering(img_crop)

    # Refine segmentation with GrabCut
    grabcut_mask = run_grabcut(img_crop, color_mask)

    # Extract features
    edge_mask = extract_edges(img_crop)
    shape_mask = extract_elongated_shapes(color_mask)

    # Select region mask
    if method == "border":
        region_mask = extract_border_regions(color_mask)
    elif method == "valid_region":
        region_mask = extract_big_regions(color_mask)
    else:
        raise ValueError("Unknown method")

    # Convert masks to probabilities
    color_prob = color_mask.astype(np.float32) / 255
    grabcut_prob = grabcut_mask.astype(np.float32) / 255
    edge_prob = edge_mask.astype(np.float32) / 255
    shape_prob = shape_mask.astype(np.float32) / 255
    region_prob = region_mask.astype(np.float32) / 255

    # Compute blur-dependent weights
    blur_score = compute_blur_score(img_crop)
    bf = blur_factor(blur_score)
    logger.debug(f"Blur score: {blur_score:.2f}, blur factor: {bf:.2f}")
    w_color, w_grabcut, w_edge, w_shape = compute_dynamic_weights(bf)
    logger.debug(f"Weights = color:{w_color:.3f}, grabcut:{w_grabcut:.3f}, edge:{w_edge:.3f}, shape:{w_shape:.3f}")

    # Compute probability map
    prob_map = (
        w_color * color_prob +
        w_grabcut * grabcut_prob +
        w_edge * edge_prob +
        w_shape * shape_prob
    )

    prob_map *= region_prob

    # Threshold probability map
    final_mask = (prob_map > 0.5).astype(np.uint8) * 255

    if dataset_dir is not None and save_subdir is not None:
        img_name = os.path.basename(img_path)

        # Save cropped image
        crop_img_name = 'cropped_' + img_name
        crop_image_path = os.path.join(dataset_dir, 'cropped_images', crop_img_name)
        os.makedirs(os.path.dirname(crop_image_path), exist_ok=True)
        plt.imsave(crop_image_path, img_crop, dpi=300)

        # Define method directory
        method_dir = os.path.join(dataset_dir, save_subdir)

        # Save binary mask
        mask_name = 'bin_' + img_name
        mask_path = os.path.join(method_dir, 'binary_segmentations', mask_name)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, final_mask)

    # Show debug visualizations
    if debug:
        fig, axes = plt.subplots(2, 4, figsize=(14,7))

        axes[0,0].imshow(img_crop)
        axes[0,0].set_title("Original Image")
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
        axes[1,0].set_title("Edge mask")
        axes[1,0].axis("off")

        axes[1,1].imshow(shape_mask, cmap="gray")
        axes[1,1].set_title("Shape mask")
        axes[1,1].axis("off")

        axes[1,2].imshow(region_mask, cmap="gray")
        axes[1,2].set_title("Region mask")
        axes[1,2].axis("off")

        axes[1,3].imshow(final_mask, cmap="gray")
        axes[1,3].set_title("Final mask")
        axes[1,3].axis("off")

        plt.suptitle(f"Image processing\n\
Blur score: {blur_score:.2f} | Blur factor: {bf:.2f}\n\
Weights = color:{w_color:.3f}, grabcut:{w_grabcut:.3f}, edge:{w_edge:.3f}, shape:{w_shape:.3f}")
        plt.tight_layout()

        if dataset_dir is not None and save_subdir is not None:
            # Save debug figure
            debug_file_name = 'DEBUG_' + img_name
            debug_file_path = os.path.join(dataset_dir, save_subdir, 'DEBUG', debug_file_name)
            os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)
            plt.savefig(debug_file_path, dpi=300)
        plt.close(fig)

    return final_mask, prob_map, img_crop

def crop_image(img: np.ndarray, crop_pix_x: int, crop_pix_y: int, width: int, height: int) -> np.ndarray:
    """
    Crop an image using top-left corner and dimensions.
    """
    return img[crop_pix_y:crop_pix_y+height, crop_pix_x:crop_pix_x+width]

def color_filtering(img_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a binary mask using HSV and opponent color filtering to detect metallic tools.
    """
    # Convert RGB image to HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    h = img_hsv[:,:,0]  # Store hue values
    s = img_hsv[:,:,1]  # Store saturation values
    v = img_hsv[:,:,2]  # Store value intensities

    # Define opponent color channels
    R = img_rgb[:,:,0].astype(np.float32)  # Store red channel
    G = img_rgb[:,:,1].astype(np.float32)  # Store green channel
    B = img_rgb[:,:,2].astype(np.float32)  # Store blue channel
    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2*B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)

    # Define metallic-tool thresholds
    # Apply HSV thresholds
    mask_hue = (h >= 20) | (h <= 170)  # Remove red and orange tones
    mask_sat = s <= 70  # Keep low saturation
    mask_val = v >= 40  # Remove dark areas

    # Apply opponent-color thresholds
    mask_o1 = np.abs(O1) < 20
    mask_o2 = np.abs(O2) < 20

    # Combine masks
    mask = mask_hue & mask_sat & mask_val & mask_o1 & mask_o2
    mask = mask.astype(np.uint8) * 255

    # Apply morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def run_grabcut(img_rgb: np.ndarray, init_mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """
    Apply GrabCut segmentation initialized from a mask and return a refined binary mask.
    """
    # Prepare GrabCut mask
    grabcut_mask = np.where(init_mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')

    # Initialize GrabCut models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut refinement
    cv2.grabCut(img_rgb, grabcut_mask, None, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)

    # Convert GrabCut output
    grabcut_result = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),  # Keep foreground and probable foreground
        255,
        0
    ).astype('uint8')

    return grabcut_result

def extract_edges(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract edge features using Canny edge detection.
    """
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Apply Canny filter
    edge_mask = cv2.Canny(img_gray, 20, 200)

    return edge_mask

def extract_elongated_shapes(mask: np.ndarray) -> np.ndarray:
    """
    Extract elongated connected components from a binary mask.
    """
    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    shape_mask = np.zeros_like(mask, dtype=np.float32)

    for i in range(1, num_labels):
        # Extract region dimensions
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Keep elongated regions
        elongation = max(width, height) / (min(width, height) + 1e-5)
        if elongation > 1.5:
            shape_mask[labels == i] = 255.0

    return shape_mask

def extract_big_regions(mask: np.ndarray, min_size: int = 1000) -> np.ndarray:
    """
    Keep only connected components larger than a given size.
    """
    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Initialize output mask
    big_regions_mask = np.zeros_like(mask, dtype=np.float32)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Keep large regions
        if area >= min_size:
            big_regions_mask[labels == i] = 255.0

    return big_regions_mask

def extract_border_regions(mask: np.ndarray) -> np.ndarray:
    """
    Extract connected components that touch the image border.
    """
    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Initialize border mask
    h, w = mask.shape
    border_feature = np.zeros_like(mask, dtype=np.float32)

    for i in range(1, num_labels):
        # Extract region statistics
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # Check border contact
        touches_border = (
            x == 0 or
            y == 0 or
            (x + width) >= w-1 or
            (y + height) >= h-1
        )

        if touches_border:
            border_feature[labels == i] = 255.0

    return border_feature

def compute_blur_score(img_rgb: np.ndarray) -> float:
    """
    Compute a blur score based on the variance of the Laplacian.
    """
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)

    # Compute Laplacian variance
    blur_score = laplacian.var()
    return blur_score

def blur_factor(blur_score: float, min_blur: float = 50, max_blur: float = 150) -> float:
    """
    Normalize a blur score to a factor between 0 and 1.
    """
    factor = (blur_score - min_blur) / (max_blur - min_blur)
    factor = np.clip(factor, 0.0, 1.0)
    return factor

def compute_dynamic_weights(bf: float) -> tuple[float, float, float, float]:
    """
    Compute normalized weights for different feature maps based on blur factor.
    """
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
