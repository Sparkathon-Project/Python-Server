import json
import re
from src.constants import ECOMMERCE_CLASSES

def call_llm(prompt,client):
    """
    Calls Gemini Pro LLM with a prompt and returns the parsed JSON response.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        text = response.text.strip()
        response = safe_parse_llm_response(text)
        return response

    except Exception as e:
        return {"error": str(e)}


def safe_parse_llm_response(text):
    """
    Safely parses LLM response that may include ```json formatting.
    Returns a dictionary or raises a JSONDecodeError.
    """
    # Remove markdown-style ```json ... ```
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    
    # Try parsing JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM JSON: {e}\nRaw content: {text}")

def get_prompt(query, detections):
    q = query.strip().lower()
    detected_classes = [d['class'] for d in detections]
    prompt = f"""
    You are an intelligent e-commerce assistant. Your goal is to understand a user's query related to an image.

    User Input:
    - Query: "{q}"
    - Detected Objects: {detected_classes}

    Available Product Categories: {ECOMMERCE_CLASSES}

    **Your Task:**
    1.  **Analyze the Query:** Determine the user's primary intent. Are they looking for something that looks the same, or something that goes with an item?
    2.  **Identify Focus Object:** From the "Detected Objects" list, determine which object the user's query is about. If the query is ambiguous or doesn't relate to any detected objects, use "no item found".
    3.  **Determine Action:**
        - `show_similar`: If the user wants a visually similar product (e.g., "another one like this", "same style", "similar to").
        - `show_related`: If the user wants related items (e.g., "what battery does this take?", "what shoes go with this dress?", "what are different ingredients in this").
        - `none`: If the intent is unclear or no relevant object is found.
    4.  **Define Search Criteria:**
        - For `show_similar`, the "search" array should contain the class name of the `focus_object`.
        - For `show_related`, the "search" array should contain relevant categories from the "Available Product Categories" list.
        - For `none`, "search" should be an empty array.

    **Output Format:**
    Provide ONLY a single, minified JSON object. Do not include explanations or markdown formatting.
    {{
      "focus_object": "<The single most relevant class name from detected objects, or 'no item found'>",
      "action": "<'show_similar' | 'show_related' | 'none'>",
      "search": ["<list of relevant categories>"]
    }}
    """
    
    return prompt