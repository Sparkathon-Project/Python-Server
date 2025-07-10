import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.load_models import load_clip_model, load_FAISS, load_sam_model, load_YOLO_model, load_llm_model
from src.constants import ECOMMERCE_CLASSES

from src.crop_marked_image import crop_marked_image_func, get_main_object
from src.embedd_items import embed_image_func, embed_text_func
from src.faiss_search import find_similar_image, find_using_text
from src.object_detection import detect_objects_func
from src.llm import call_llm, get_prompt

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

clip_model, clip_preprocess, clip_device = load_clip_model()
faiss_index, faiss_ids_images, faiss_ids_text = load_FAISS()
sam_model, sam_predictor, mask_generator = load_sam_model()
yolo_model = load_YOLO_model()
llm_client = load_llm_model()

# Get the database connection string from environment variables
DB_CONN_STRING = os.getenv("DB_CONN_STRING")

@app.route('/querySearch', methods=['POST'])
def detect_objects():
    image_file = request.files.get('image')
    query = request.form.get('query')

    if not image_file:
        return jsonify({"error": "No image file provided"}), 400
    if not query:
        return jsonify({"error": "No query provided"}), 400
    if not image_file.mimetype.startswith('image/'):
        return jsonify({"error": "Uploaded file is not an image"}), 400

    try:
        detections = detect_objects_func(yolo_model, image_file, ECOMMERCE_CLASSES)
        prompt = get_prompt(query, detections)
        response = call_llm(prompt, llm_client)

        if response["action"] == "show_similar":
            image_bytes = get_main_object(image_file, detections, response["search"][0])
            embedding = embed_image_func(clip_model, clip_preprocess, clip_device, image_bytes)
            similar_ids = find_similar_image(faiss_index, embedding, 5, faiss_ids_images)
            return jsonify({"results": similar_ids})
        else:
            embeddings =  embed_text_func(clip_model, clip_preprocess, clip_device, response["search"])
            results = find_using_text(faiss_index, embeddings, 5, faiss_ids_text)
            return jsonify({"results": results})
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
        buf = crop_marked_image_func(sam_predictor, image_file, clicks)
        embedding = embed_image_func(clip_model, clip_preprocess, clip_device, buf).astype('float32').reshape(1, -1)
        similar_ids = find_similar_image(faiss_index, embedding, 5, faiss_ids_images)
        return jsonify({"results": similar_ids})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)