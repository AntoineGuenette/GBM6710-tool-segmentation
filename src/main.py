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
    mask_sat  = s < 60 # low saturation
    # In Opponent Color space
    mask_o1 = np.abs(O1) < 15
    mask_o2 = np.abs(O2) < 15

    # Combine masks
    mask =  mask_hue & mask_sat & mask_o1 & mask_o2
    mask = mask.astype(np.uint8) * 255

    # Morphologic filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
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
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    return grabcut_result

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
    img_crop = crop_image(img_rgb, 320, 28, 1280, 1024)

    # Define possible tool areas
    init_mask = color_filtering(img_crop)

    # Refine segmentation with GrabCut
    grabcut_result = run_grabcut(img_crop, init_mask)
    
    # Show images
    fig, axes = plt.subplots(1, 3, figsize=(12,4))

    axes[0].imshow(img_crop)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(init_mask, cmap="gray")
    axes[1].set_title("Filtrage des couleurs")
    axes[1].axis("off")

    axes[2].imshow(grabcut_result, cmap="gray")
    axes[2].set_title("Résultat GrabCut")
    axes[2].axis("off")

    plt.show()
