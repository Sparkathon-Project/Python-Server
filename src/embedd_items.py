import torch
from PIL import Image
import numpy as np

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

def embed_text_func(model, processor, device, categories):
    inputs = processor(text=categories, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs).cpu().numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings