import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def load_clip_model():
    """Loads the CLIP model and preprocessing transform."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",use_fast=True)
        return model, processor, device
    except Exception as e:
        raise RuntimeError("Could not load CLIP model. Error: {e}")

def embed_image_func(model, preprocess, device, image_buffer):
    """
    Embeds an image using a pre-loaded CLIP model.

    Args:
        model: The pre-loaded CLIP model.
        preprocess: The CLIP preprocessing transform.
        device (str): The device to run the model on.
        image_buffer (io.BytesIO): A BytesIO buffer containing the image data.

    Returns:
        torch.Tensor: The image embedding.
    """
    # Open the image from the buffer
    try:
        image = Image.open(image_buffer).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open or convert image from buffer: {e}")

    # Preprocess the image
    inputs = preprocess(images=image, return_tensors="pt").to(device)

    # Generate the embedding
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    return embedding / np.linalg.norm(embedding)