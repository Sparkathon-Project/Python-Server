import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.initial_load import load_clip_model, load_FAISS, load_sam_model, load_YOLO_model
from src.constants import CATEGORIES, ECOMMERCE_CLASSES

from src.normalise_image import normalise_image_func
from src.embedd_image import embed_image_func, classify_image_func
from src.similarity_search import find_similar_embeddings
from src.object_detection import detect_objects_func

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

clip_model, clip_preprocess, clip_device = load_clip_model()
faiss_index, faiss_ids = load_FAISS()
sam_model, sam_predictor, mask_generator = load_sam_model()
yolo_model = load_YOLO_model()

# Get the database connection string from environment variables
DB_CONN_STRING = os.getenv("DB_CONN_STRING")

@app.route('/detect-objects', methods=['POST'])
def detect_objects():
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    try:
        detections= detect_objects_func(yolo_model, image_file, ECOMMERCE_CLASSES)
        return jsonify(detections)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/search', methods=['POST'])
def search():
    image_file = request.files.get('image')
    clicks = request.form.get('clicks')

    if not image_file:
        return jsonify({"error": "No image file provided"}), 400
    if not clicks:
        return jsonify({"error": "No clicks provided"}), 400

    try:
        # Normalise the image
        buf = normalise_image_func(sam_predictor, image_file, clicks)
        # Embed the normalised image
        embedding = embed_image_func(clip_model, clip_preprocess, clip_device, buf).astype('float32').reshape(1, -1)
        # Find similar embeddings
        similar_ids = find_similar_embeddings(faiss_index, embedding, 5, faiss_ids)
        return jsonify({"results": similar_ids})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
