from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np
from PIL import Image
import io

def load_sam_model(model_type="vit_b", checkpoint="./sam_vit_b.pth"):
    """Loads the SAM model and returns the predictor."""
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    return predictor

def normalise_image_func(predictor, image_file, clicks_str):
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