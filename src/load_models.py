from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import json
import faiss
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
from google import genai

def load_sam_model():
    """Loads the SAM model and returns the predictor."""
    try:
        sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b.pth")
        sam.to("cuda" if torch.cuda.is_available() else "cpu")
        sam_predictor = SamPredictor(sam)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return sam, sam_predictor, mask_generator
    except Exception as e:
        raise RuntimeError("Could not load SAM model. Error: {e}")

def load_FAISS():
    """Loads the FAISS Index and map.json for similarity search."""
    try:
        faiss_index = faiss.read_index("./embeddings.index")
        with open("./clip_images_id_map.json", "r") as f:
            faiss_ids_images = json.load(f)
        with open("./clip_text_id_map.json", "r") as f:
            faiss_ids_text = json.load(f)
        return faiss_index, faiss_ids_images, faiss_ids_text
    except Exception as e:
        raise RuntimeError("Could not load FAISS store. Error: {e}")

def load_clip_model():
    """Loads the CLIP model and preprocessing transform."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast = True)
        return model, processor, device
    except Exception as e:
        raise RuntimeError("Could not load CLIP model. Error: {e}")

def load_YOLO_model():
    """Loads the YOLO model and Yolo classes."""
    try:
        # Load YOLO model once
        model = YOLO("yolov8m.pt")
        return model
    except Exception as e:
        raise RuntimeError("Could not load YOLO model. Error: {e}")

def load_llm_model():
    client = genai.Client()
    return client