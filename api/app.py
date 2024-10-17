import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from jsonschema import validate, ValidationError
import re
from flask_caching import Cache
import validators
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()

# Access the API keys
PERPLEXITY_API = os.getenv("PERPLEXITY_API")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
EBAY_ID = os.getenv("EBAY_ACCESS_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"

# Configure caching
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Use a ThreadPoolExecutor for concurrent API calls
executor = ThreadPoolExecutor(max_workers=10)

# Define the expected JSON schema for validation
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
                    "website": {"type": "string"},
                    "product_image_url": {"type": "string"},
                },
                "required": ["supplier", "product_name", "price", "location", "email"]
            }
        }
    },
    "required": ["suppliers"]
}


def query_perplexity(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API}",
        "Content-Type": "application/json"
    }

    schema_description = """
    Provide a JSON response with the following structure:
    {
        "suppliers": [
            {
                "supplier": "string",
                "product_name": "string",
                "price": "string",
                "location": "string",
                "supplier_contact": "string",
                "email": "string",
                "website": "string",
                "product_image_url": "string"
            }
        ]
    }
    """

    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {"role": "system", "content": f"Generate a response using the following schema: {schema_description}. "
                "Ensure your response only contains JSON matching this structure."},
            {"role": "user", "content": f"{prompt} prioritize results with actual price, supplier's name, product name, and location as mandatory fields; return at least four results"}
        ],
        "temperature": 0
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=100)
    response.raise_for_status()

    return response.json()  # We return JSON


def query_general(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
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
        response = requests.post(API_URL, headers=headers, json=payload, timeout=100)
        response.raise_for_status()

        response_data = response.json()

        if 'choices' not in response_data or not response_data['choices']:
            return {"error": "No valid response from the API."}

        content = response_data['choices'][0].get('message', {}).get('content', "")

        if not content:
            return {"error": "Empty content returned from the API."}

        # Clean the content by removing unnecessary characters
        cleaned_content = re.sub(r"[#*]+", "", content).strip()

        # Keep it short, return the content as-is
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


def detect_product_query(prompt):
    """Detects whether the prompt is related to product price requests."""
    keywords = ['price', 'product', 'cost', 'buy', 'purchase']
    return any(keyword in prompt.lower() for keyword in keywords)


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



@app.route('/product_prices', methods=['POST'])
def get_product_prices():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in the request body"}), 400

        prompt = data.get("prompt").strip()
        limit = data.get("limit", 10)

        print(f"Received prompt: {prompt} with limit: {limit}")  # Debugging statement

        if detect_product_query(prompt):
            # Product-related query, proceed to query structured data
            future = executor.submit(query_perplexity, f"{prompt} return {limit} results")
            response_dict = future.result()

            # print(f"Response from Perplexity: {response_dict}")  # Debugging statement

            # Extract the 'choices' field from the Perplexity response
            choices = response_dict.get("choices", [])
            if not choices:
                return jsonify({"error": "Sorry, I cannot find sufficient data to respond to your request."}), 500

            # Extract the JSON content from the Perplexity response
            content = choices[0].get("message", {}).get("content", "")
            json_data = extract_json_from_response(content)

            if not json_data:
                return jsonify({"error": "Sorry, I cannot find sufficient data to respond to your request."}), 500

            # Validate against the expected schema
            validate(instance=json_data, schema=expected_schema)

            # Extract the number of results and the first product's details for the introductory message
            suppliers = json_data.get("suppliers", [])
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

                # Check if the product image URL is valid and not empty; if not, use placeholder image
                if not product_image_url or not validators.url(product_image_url):
                    product_image_url = "/static/placeholder.png"

                structured_response.append({
                    "supplier": item.get("supplier", "Unknown Supplier"),
                    "product_name": product_name,
                    "price": price,
                    "location": item.get("location", "Unknown Location"),
                    "supplier_contact": item.get("supplier_contact", "Not Available"),
                    "email": item.get("email", "Not Available"),
                    "website": item.get("website", "Not Available"),
                    "product_image_url": product_image_url
                })

            # Return the response with the introductory message
            return jsonify({
                "message": intro_message,
                "suppliers": structured_response
            })

        else:
            try:
                future = executor.submit(query_general, prompt)
                response_dict = future.result()

                # Debug: Print response_dict for inspection
                print("General Query Response:", response_dict)

                # Check if there was an error in the API response
                if 'error' in response_dict:
                    return jsonify(response_dict), 500

                # Extract cleaned content
                cleaned_content = response_dict.get("response", "Sorry, I cannot provide a valid response.")

                return jsonify({"response": cleaned_content})

            except Exception as e:
                print(f"An unexpected error occurred: {e}")  # Log unexpected errors
                return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
    except json.JSONDecodeError as json_error:
        print(f"JSON decode error: {json_error}")  # Log JSON decode errors
        return jsonify({"error": "JSON decoding error", "details": str(json_error)}), 500

    except ValidationError as val_error:
        return jsonify({"error": "JSON schema validation error", "details": str(val_error)}), 500

    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # Log unexpected errors
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, port=5000)
