import io
import os

import numpy as np
from PIL import Image

from src.object_detection import get_best_detections

def crop_marked_image_func(predictor, image_file, clicks_str):
    """
    Normalises an image using a pre-loaded SAM predictor.

    Args:
        predictor (SamPredictor): The pre-loaded SAM predictor.
        image_file: The image file to process.
        clicks_str (str): A string representation of the clicks.

    Returns:
        io.BytesIO: A buffer containing the normalised image.
    """
    try:
        # Safely evaluate clicks string into a numpy array
        clicks_list = eval(clicks_str)
        if not isinstance(clicks_list, list):
            raise ValueError("Clicks data is not a list.")
        clicks = np.array(clicks_list, dtype=np.int32)
    except Exception as e:
        raise ValueError(f"Invalid clicks format: {e}")

    if len(clicks) == 0:
        raise ValueError("Empty click list")

    # positive clicks are labeled with 1
    labels = np.ones(len(clicks))

    # Open image using PIL
    try:
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)
    except Exception as e:
        raise ValueError(f"Could not open or convert image: {e}")

    predictor.set_image(image_np)

    masks, _, _ = predictor.predict(
        point_coords=clicks,
        point_labels=labels,
        multimask_output=False
    )

    mask = masks[0]
    
    # This reduces the image dimensions to only include the object and its minimal transparent border
    y, x = np.where(mask)
    if y.size > 0 and x.size > 0: # Check if mask is not empty
        top, bottom, left, right = y.min(), y.max(), x.min(), x.max()
        cropped = image_np[top:bottom+1, left:right+1]
        result_image = Image.fromarray(cropped)
    else:
        # If the mask is empty, return an empty transparent image
        transparent_image_np = np.zeros((1, 1, 4), dtype=np.uint8)
        result_image = Image.fromarray(transparent_image_np, 'RGBA')


    # Save the result to a BytesIO object and send
    buf = io.BytesIO()
    result_image.save(buf, format='PNG')
    buf.seek(0)
    
    return buf

import io

def get_main_object(image_file, detections, search_category):
    pil_image = Image.open(image_file).convert("RGB")
    detection = get_best_detections(detections, search_category)

    if detection is None:
        raise ValueError(f"No matching detection found for category: {search_category}")

    bbox = detection["bbox"]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    cropped_image = pil_image.crop((x1, y1, x2, y2))

    image_bytes = io.BytesIO()
    cropped_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    return image_bytes
