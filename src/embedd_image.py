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

def classify_image_func(model, processor, device, image_embedding, categories):
    """
    Classifies image embedding against category text using CLIP.

    Args:
        model: CLIPModel instance.
        processor: CLIPProcessor instance.
        device: 'cpu' or 'cuda'.
        image_embedding: np.ndarray (normalized image embedding).
        categories: list of category strings.

    Returns:
        str: Most likely category label.
    """
    # Tokenize and encode text
    text_inputs = processor(text=categories, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    # Normalize and convert image embedding
    image_tensor = torch.tensor(image_embedding).to(device)
    image_tensor = image_tensor / image_tensor.norm()

    # Compute cosine similarity
    logits = image_tensor @ text_embeddings.T
    probs = logits.softmax(dim=0)

    return categories[torch.argmax(probs).item()]
