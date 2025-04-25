# JSRT Dataset Utility Functions
# This file contains utility functions for processing the JSRT dataset

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Constants
IMG_HEIGHT = 512
IMG_WIDTH = 512
MASK_SUBDIRS = ['heart', 'left_clavicle', 'left_lung', 'right_clavicle', 'right_lung']

def read_img_file(file_path):
    """
    Read JSRT .IMG files which are 2048x2048 grayscale images
    stored as binary data.
    
    Parameters:
    -----------
    file_path : str
        Path to the .IMG file
        
    Returns:
    --------
    numpy.ndarray
        Normalized image array with values in [0, 1]
    """
    try:
        # JSRT dataset stores images as 2048x2048 pixels, 12 bits/pixel
        # Stored as unsigned short (2 bytes per pixel)
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
        
        # Reshape to 2D (2048x2048)
        img = data.reshape(2048, 2048)
        
        # Normalize to [0, 1]
        img = img / np.max(img)
        
        return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_mask(img_id, mask_dir, mask_subdirs=MASK_SUBDIRS):
    """
    Load segmentation masks for a given image ID and combine them.
    
    Parameters:
    -----------
    img_id : str
        Image ID (filename without extension)
    mask_dir : str
        Base directory containing segmentation masks
    mask_subdirs : list
        List of subdirectories, each containing masks for different organs
        
    Returns:
    --------
    numpy.ndarray
        Combined mask with shape (height, width, num_masks)
    """
    masks = []
    
    for mask_subdir in mask_subdirs:
        mask_path = os.path.join(mask_dir, mask_subdir, f"{img_id}.png")
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            mask = mask / 255.0  # Normalize to [0, 1]
        else:
            # If mask doesn't exist, create an empty one
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
        
        masks.append(mask)
    
    # Stack masks into multi-channel array
    combined_mask = np.stack(masks, axis=-1)
    return combined_mask

def resize_img(img, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Resize image to target size.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    target_size : tuple
        Target size as (height, width)
        
    Returns:
    --------
    numpy.ndarray
        Resized image
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def load_clinical_data(base_dir):
    """
    Load clinical data from CLNDAT_EN.txt and CNNDAT_EN.txt files.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing the 'clinical' folder
        
    Returns:
    --------
    dict, dict
        Dictionaries containing nodule and non-nodule data
    """
    nodule_data = {}
    non_nodule_data = {}
    
    # Parse CLNDAT_EN.txt (nodule data)
    try:
        with open(os.path.join(base_dir, "clinical", "CLNDAT_EN.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:  # Ensure we have enough parts to extract coordinates
                    img_id = parts[0].split('.')[0]  # Remove .IMG extension
                    try:
                        # The coordinates are in the 5th and 6th positions
                        x_coord = int(parts[5])
                        y_coord = int(parts[6])
                        
                        # Determine nodule size if available
                        # In the example, it seems nodule size might be in position 2
                        nodule_size = float(parts[2]) if len(parts) > 2 else 0
                        
                        nodule_data[img_id] = {
                            'label': 1,  # 1 for nodule
                            'nodule_size': nodule_size,
                            'x_coord': x_coord,
                            'y_coord': y_coord
                        }
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing nodule data for {img_id}: {e}")
                        continue
    except Exception as e:
        print(f"Error loading nodule clinical data: {e}")
    
    # Parse CNNDAT_EN.txt (non-nodule data)
    try:
        with open(os.path.join(base_dir, "clinical", "CNNDAT_EN.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    img_id = parts[0].split('.')[0]  # Remove .IMG extension
                    non_nodule_data[img_id] = {
                        'label': 0  # 0 for non-nodule
                    }
    except Exception as e:
        print(f"Error loading non-nodule clinical data: {e}")
    
    return nodule_data, non_nodule_data

def plot_image_with_masks(img, masks, title=None, nodule_coords=None):
    """
    Plot an image with its segmentation masks overlaid.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    masks : numpy.ndarray
        Segmentation masks with shape (height, width, num_masks)
    title : str, optional
        Plot title
    nodule_coords : tuple, optional
        (x, y) coordinates of nodule if present
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Plot each mask on a separate subplot
    mask_names = ['Heart', 'Left Clavicle', 'Left Lung', 'Right Clavicle', 'Right Lung']
    
    for i, (mask_name, mask) in enumerate(zip(mask_names, [masks[..., i] for i in range(masks.shape[-1])])):
        row, col = (i+1) // 3, (i+1) % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].imshow(mask, cmap='jet', alpha=0.4)
        axes[row, col].set_title(mask_name)
        if nodule_coords is not None:
            x, y = nodule_coords
            # Scale coordinates to match resized images
            x_scaled = int(x * img.shape[1] / 2048)
            y_scaled = int(y * img.shape[0] / 2048)
            axes[row, col].scatter(x_scaled, y_scaled, c='red', marker='x', s=100)
        axes[row, col].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()

def plot_with_attention_mask(img, attention_mask, title=None, nodule_coords=None):
    """
    Plot an image with an attention mask and nodule location.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    attention_mask : numpy.ndarray
        Attention mask highlighting nodule region
    title : str, optional
        Plot title
    nodule_coords : tuple, optional
        (x, y) coordinates of nodule if present
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Image with attention mask
    axes[1].imshow(img, cmap='gray')
    axes[1].imshow(attention_mask.squeeze(), cmap='jet', alpha=0.4)
    axes[1].set_title("Attention Mask")
    if nodule_coords is not None:
        x, y = nodule_coords
        # Scale coordinates to match resized images
        x_scaled = int(x * img.shape[1] / 2048)
        y_scaled = int(y * img.shape[0] / 2048)
        axes[1].scatter(x_scaled, y_scaled, c='red', marker='x', s=100)
    axes[1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()