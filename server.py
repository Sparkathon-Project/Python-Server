import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.normalise_image import load_sam_model, normalise_image_func
from src.embedd_image import load_clip_model, embed_image_func
from src.similarity_search import find_similar_embeddings , load_FAISS 
# from src.db_search import fetch_products_by_ids 

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load models on startup
sam_predictor = load_sam_model()
clip_model, clip_preprocess, clip_device = load_clip_model()
faiss_index, faiss_ids = load_FAISS()

# Get the database connection string from environment variables
DB_CONN_STRING = os.getenv("DB_CONN_STRING")

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
        """
        products = fetch_products_by_ids(similar_ids, DB_CONN_STRING)
        results = []
        for item in products:
            results.append({
                "id": item[0],
                "title": item[1],
                "price": float(item[2]) if item[2] is not None else None,
                "image": item[3],
                "description": item[4],
                "miscellaneous": item[5]
            })
        """
        return jsonify({"results": similar_ids})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)