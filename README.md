## Prerequisites

Before you begin, ensure you have the following installed:

- Python
- Pip (Python package installer)
- Git

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Sparkathon-Project/Python-Server.git
    cd Backend
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    ```bash
    .\.venv\Scripts\activate
    ```

4.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the required models and data:**

    The following files are required for the application to run. Please download them and place them in the root directory of the project (`Backend/`):

    -   `sam_vit_b.pth`: The Segment Anything Model (SAM) checkpoint. You can download this from the [official repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).
    -   `embeddings.index`: The FAISS index file.
    -   `clip_id_map.json`: The JSON file mapping FAISS indices to product IDs.

6.  **Set up environment variables:**

    Create a `.env` file in the root directory of the project and add the following line:

    ```
    DB_CONN_STRING="your-database-connection-string"
    ```

    Replace `"your-database-connection-string"` with the actual connection string for your PostgreSQL database.

## Running the Server

Once you have completed the setup steps, you can run the server with the following command:

```bash
python server.py
```

The server will start on `http://0.0.0.0:5000`.

## API Endpoints

### `/search`

-   **Method:** `POST`
-   **Description:** Performs a visual search based on an input image and user clicks.
-   **Form Data:**
    -   `image`: The input image file.
    -   `clicks`: A JSON string representing the user's clicks on the image (e.g., `[[x1, y1], [x2, y2]]`).
-   **Success Response:**
    -   **Code:** 200
    -   **Content:** `{"results": [id1, id2, ...]}`
-   **Error Response:**
    -   **Code:** 400
    -   **Content:** `{"error": "Error message"}`
