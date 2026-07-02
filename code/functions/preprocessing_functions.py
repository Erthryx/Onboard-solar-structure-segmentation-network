import numpy as np
from PIL import Image
from datetime import datetime

def find_304_match(target_str, date_obs):
    target_dt = datetime.strptime(target_str, "%Y_%m_%d__%H_%M_%S_%f")
    fmt = "%Y-%m-%dT%H:%M:%S.%f"

    zarr_times = [datetime.strptime(t, fmt) for t in date_obs]

    # Compute the time difference in seconds
    diffs = np.abs(np.array([(t - target_dt).total_seconds() for t in zarr_times]))

    closest_index = diffs.argmin()
    closest_time = zarr_times[closest_index]

    # print("Target time:", target_dt)
    # print("Closest DATE-OBS:", closest_time)
    # print("Closest index:", closest_index)
    # print("Difference in seconds:", diffs[closest_index])
    return closest_time, diffs[closest_index], closest_index



def normalizing_data(data, percentile_top_limite=99, mask = np.array(Image.open(r"C:\Users\User\Master_physics\Master_Thesis\addittional_dataset\test_limb_mask.png").convert("L"))):
    """
    data: np.array (#,256,256)
    mask: np.array (256,256) where 1 = inside sun, 0 = background
    return: np.array of normalized ([0,1]) images   
    """
    if type(percentile_top_limite) != list:
        percentile_top_limite = [percentile_top_limite] * len(data)
    
    norm_data = []

    # Flatten mask only once
    mask_flat = mask.astype(bool)

    for i in range(len(data)):
        img = data[i]

        # Apply mask: remove background values
        img_inside = img[mask_flat]

        # Compute percentiles ONLY on sun pixels
        vmin, vmax = np.percentile(img_inside, [0.1, percentile_top_limite[i]])

        # Clip and normalize your FULL image (keeping background)
        clipped_image = np.clip(img, vmin, vmax)
        norm_image = (clipped_image - vmin) / (vmax - vmin)

        # Restore background to 1 (or 0 if you prefer)
        norm_image[~mask_flat] = 1  

        norm_data.append(norm_image)

    return np.array(norm_data)



def crop_data(data, mask = np.array(Image.open(r"C:\Users\User\Master_physics\Master_Thesis\addittional_dataset\test_limb_mask.png").convert("L"))):
    """
    data: list/np.array (#,256,256)
    return: np.array of images where a circular mask was applied to set to 0 all the pixels outside the sun
    """

    cropped_data = []
    for image in data:
        bg_color = 1
        print(mask.shape)
        assert image.shape == mask.shape, "Image and mask must have the same shape"
        inside_circle = mask == 255
        cropped = image.copy()
        cropped[~inside_circle] = bg_color
        cropped_img = cropped
        cropped_data.append(cropped_img)
    cropped_data = np.array(cropped_data)

    return cropped_data

def crop_single_data(data, mask = np.array(Image.open(r"C:\Users\User\Master_physics\Master_Thesis\addittional_dataset\test_limb_mask.png").convert("L")) ,is_black_not_white = True, norm_top_val = 255):
    """
    data: list/np.array (#,256,256)
    return: np.array of images where a circular mask was applied to set to 0 all the pixels outside the sun
    """
    if is_black_not_white:
        bg_color = 0
    else:
        bg_color = norm_top_val
    assert data.shape == mask.shape, "Image and mask must have the same shape"
    inside_circle = mask == 255
    cropped = data.copy()
    cropped[~inside_circle] = bg_color
    cropped_img = cropped

    return cropped_img

import numpy as np
from PIL import Image

def normalize_single_data(img, percentile_top_limit=99.5, mask=np.array(Image.open(r"C:\Users\User\Master_physics\Master_Thesis\addittional_dataset\test_limb_mask.png").convert("L")), norm_top_val = 1):
    """
    img: np.array (256,256)
    mask: np.array (256,256) where 1 = inside sun, 0 = background
    return: normalized image ([0,1])
    """

    mask_flat = mask.astype(bool)

    # Extract only sun pixels
    img_inside = img[mask_flat]

    # Compute percentiles on valid pixels
    vmin, vmax = np.percentile(img_inside, [0.1, percentile_top_limit])

    # Clip + normalize full image
    clipped_image = np.clip(img, vmin, vmax)
    norm_image = (clipped_image - vmin) / (vmax - vmin)

    if norm_top_val == 255:
        norm_image = (norm_image * 255).astype(np.uint8)

    # Set background
    norm_image[~mask_flat] = norm_top_val

    return norm_image

