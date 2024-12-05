import os
import json
import requests
from flask import  jsonify,url_for
from jsonschema import validate, ValidationError
import re
import validators
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor


# Access the API keys
EBAY_ID = os.getenv("EBAY_ACCESS_KEY")
OPENAI_API_URL ="https://api.openai.com/v1/chat/completions"
MONGO_URI = os.getenv("MONGO_URI")
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")


expected_schema = {
    "type": "object",
    "properties": {
        "suppliers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "supplier": {"type": "string"},
                    "product_name": {"type": "string"},
                    "price": {"type": "string"},
                    "location": {"type": "string"},
                    "supplier_contact": {"type": "string"},
                    "email": {"type": "string"},
                    "product_url": {"type": "string"},
                    "product_image_url": {"type": "string"},
                },
                "required": ["supplier", "product_name", "price", "location", "product_url","email"]
            }
        }
    },
    "required": ["suppliers"]
}



# ThreadPoolExecutor for concurrent API calls
executor = ThreadPoolExecutor(max_workers=10)

def query_openai(prompt):
    headers = {
        "Authorization": f"Bearer {OPEN_AI_KEY}",
        "Content-Type": "application/json"
    }
    schema_description = """
Please provide a JSON response with the following structure:
{
    "suppliers": [
        {
            "supplier": "string",  # Supplier name
            "product_name": "string",  # Name of the product
            "price": "string",  # Current market price in the local currency
            "location": "string",  # Location of the supplier
            "supplier_contact": "string",  # Supplier's contact name
            "email": "string",  # Supplier's valid email address
            "product_url": "string",  # Product's official URL (clickable)
            "product_image_url": "string"  # URL for the product image
        }
    ]
}

Please ensure that:
1. Only top-rated suppliers with high ratings (if available) are included in the response.
2. Prices are reflective of the current market rates.
3. The product URL and email addresses must be active, valid, and functional.
4. The response contains only valid JSON according to the structure above.
"""

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": f" your are a product information speacilist. Generate a response using the following schema: {schema_description}. "
                "Ensure your response only contains JSON matching this structure."},
            {"role": "user", "content": f"{prompt}"}
        ],
        "temperature": 0
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=100)
    response.raise_for_status()
    return response.json()

def extract_json_from_response(content):
    """ Extract and clean JSON from markdown """
    try:
        json_match = re.search(r'```json(.*?)```', content, re.DOTALL)
        if json_match:
            clean_content = json_match.group(1).strip()
            return json.loads(clean_content)
        else:
            raise ValueError("No JSON found in the response")
    except (json.JSONDecodeError, ValueError) as e:
        return None, str(e)

def get_product_price_data(prompt, limit):
    try:
        future = executor.submit(query_openai, f"{prompt} return {limit} results")
        response_dict = future.result()
        print(response_dict)
        # Extract the 'choices' field from the Perplexity response
        choices = response_dict.get("choices", [])
        if not choices:
            return jsonify({"error": "Sorry, I cannot find sufficient data to respond to your request."}), 500
            # Extract the JSON content from the Perplexity response
        content = choices[0].get("message", {}).get("content", "")
            # json_data = extract_json_from_response(content)
        json_data = json.loads(content)
        if not json_data:
                print("there is an error 1")
                return jsonify({"error": "Sorry, I cannot find sufficient data to respond to your request."}), 500

            # Validate against the expected schema
        validate(instance=json_data, schema=expected_schema)

            # Extract the number of results and the first product's details for the introductory message
        suppliers = json_data["suppliers"]

        num_results = len(suppliers)
        if num_results == 0:
            return jsonify({"error": "No suppliers found."}), 404
        # Use the first supplier's product name and location for the message
        first_supplier = suppliers[0]
        product_name = first_supplier.get("product_name", "the product")
        location = first_supplier.get("location", "the specified location")
        # Form the introductory message
        intro_message = f"I have found {num_results} results for {product_name} in {location}."
        # Prepare the structured response
        structured_response = []
        for item in suppliers:
                    product_name = item.get("product_name", "Unknown Product")
                    price = item.get("price", "Price not found")
                    product_image_list = get_ebay_product_images(product_name, EBAY_ID, limit)
                    product_image_url = product_image_list[0] if product_image_list else "Image not Available"
                    if not product_image_url or not validators.url(product_image_url):
                    # Use url_for to generate correct static file path
                        product_image_url = url_for('static', filename='placeholder.png')
                    # # Check if the product image URL is valid and not empty; if not, use placeholder image
                    # if not product_image_url or not validators.url(product_image_url):
                    #     product_image_url = "/static/placeholder.png"
                    structured_response.append({
                        "supplier": item.get("supplier", "Unknown Supplier"),
                        "product_name": product_name,
                        "price": price,
                        "location": item.get("location", "Unknown Location"),
                        "supplier_contact": item.get("supplier_contact", "Not Available"),
                        "email": item.get("email", "Not Available"),
                        "product_url": item.get("product_url", "Not Available"),
                        "product_image_url": product_image_url
                    })
                # Return the response with the introductory message
        return jsonify({
                    "message": intro_message,
                    "suppliers": structured_response
                })
    except ValidationError as e:
        # Handle JSON schema validation errors
        return jsonify({"error": "Data validation failed", "details": str(e)}), 400
    except requests.exceptions.HTTPError as http_err:
        # Log error details for debugging
        print(f"HTTP error occurred: {http_err}")
        return jsonify({"error": "Service unavailable. Please try again later."}), 503
    except Exception as e:
        # Capture any other exceptions for debugging
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "Internal server error. Please contact support."}), 500


def get_ebay_product_images(product_name, ebay_app_id, num_results):
    api_url = "https://svcs.ebay.com/services/search/FindingService/v1"
    params = {
        "OPERATION-NAME": "findItemsByKeywords",
        "SERVICE-VERSION": "1.0.0",
        "SECURITY-APPNAME": ebay_app_id,
        "RESPONSE-DATA-FORMAT": "JSON",
        "keywords": product_name,
        "paginationInput.entriesPerPage": num_results,
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get("findItemsByKeywordsResponse", [{}])[0].get("searchResult", [{}])[0].get("item", [])
        return [item.get("galleryURL", ["No Image"])[0] for item in items]
    except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as e:
        return f"Error fetching images: {e}"

def detect_product_query(prompt):
    """Detects whether the prompt is related to product price requests."""
    keywords = ['price', 'product', 'cost', 'buy', 'purchase']
    return any(keyword in prompt.lower() for keyword in keywords)
def query_general(prompt):
    headers = {
        "Authorization": f"Bearer {OPEN_AI_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a friendly and conversational assistant. Provide a **very short summary**, preferably in one or two sentences."
            },
            {
                "role": "user",
                "content": f"{prompt} Provide a very short and concise summary in no more than two sentences."
            }
        ],
        "temperature": 0.2  # Keep a lower temperature for more deterministic responses
    }
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=100)
        response.raise_for_status()
        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            return {"error": "No valid response from the API."}
        content = response_data['choices'][0].get('message', {}).get('content', "")
        if not content:
            return {"error": "Empty content returned from the API."}
        # Clean the content by removing unnecessary characters
        cleaned_content = re.sub(r"[#*]+", "", content).strip()
        return {"response": cleaned_content}
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return {"error": "HTTP error occurred", "details": str(http_err)}
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return {"error": "Request error occurred", "details": str(req_err)}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": "An unexpected error occurred", "details": str(e)}
