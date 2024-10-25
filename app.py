import os
import json
import requests
from flask import Flask, request, jsonify,url_for
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from jsonschema import validate, ValidationError
import re
from flask_caching import Cache
import validators
from flask_cors import CORS
from pymongo import MongoClient
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.llms import OpenAI
import uuid
from datetime import datetime, timezone

app = Flask(__name__, static_folder='static')
app = Flask(__name__)
CORS(app)



load_dotenv()



# Access the API keys
PERPLEXITY_API = os.getenv("PERPLEXITY_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
EBAY_ID = os.getenv("EBAY_ACCESS_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"
CHAT_QUERY_URL = "https://api.perplexity.ai/query"
MONGO_URI = os.getenv("MONGO_URI")

print(PERPLEXITY_API)

# Use a ThreadPoolExecutor for concurrent API calls
executor = ThreadPoolExecutor(max_workers=10)

client = MongoClient(MONGO_URI)
db = client.sourcify
purchase_requests_collection = db.requests
# sample = purchase_requests_collection.find_one()
# print(sample)


# # List collection names

# print(list(purchase_requests_collection.find({})))

# Configure caching
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Session data store to track conversation state
session_data = {}
def generate_pr_id():
    return f"PR-{uuid.uuid4().hex[:6]}"

# Function to create dynamic prompts based on conversation state
def create_prompt(current_step, user_input=None):
    if current_step == 'start':
        return "Hello! Let's create a new purchase request. When is the due date for the request?"
    elif current_step == 'due_date':
        return f"Got it! The due date is {user_input}. What is the reason for this purchase?"
    elif current_step == 'reason':
        return f"Reason noted: {user_input}. Now, what is the priority of this request? (Urgent, High, Medium, Low)"
    elif current_step == 'priority':
        return f"Priority set to {user_input}. Please list the items you need and their quantities, e.g., '50 desks, 30 monitors'."
    elif current_step == 'items':
        return f"Items recorded: {user_input}. Would you like to submit this request now?"
    elif current_step == 'another_request':
        return "Would you like to create another request? Type 'yes' to start over or 'no' to exit."
    return "I didn't understand your input. Please try again."

# Function to validate input for each step
def validate_input(current_step, user_input):
    if current_step == "due_date":
        try:
            due_date = datetime.strptime(user_input, '%Y-%m-%d')
            current_date = datetime.now().date()  # Get today's date
            if due_date.date() < current_date:
                return False, "The due date cannot be in the past. Please provide a valid date (today or later)."
        except ValueError:
            return False, "Please enter the due date in YYYY-MM-DD format."

    elif current_step == "priority":
        if user_input not in ["Urgent", "High", "Medium", "Low"]:
            return False, "Priority must be one of Urgent, High, Medium, or Low."

    return True, None

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')  # Retrieve user_id from request to track session
    user_input = request.json.get('message')  # Retrieve user's input

    # Initialize session if it's a new user or reset session state if empty
    if user_id not in session_data or 'step' not in session_data[user_id]:
        session_data[user_id] = {
            "step": "start",  # Track current step in the conversation
            "due_date": None,
            "reason": None,
            "priority": None,
            "items": None
        }

    session = session_data[user_id]  # Access session data for the current user
    current_step = session['step']

    # Start of conversation
    if current_step == "start":
        session['step'] = "due_date"  # Move to the next step
        return jsonify({"response": "Hello! Let's create a purchase request. When is the due date for the request?"})

    # Validate input if necessary
    is_valid, error_msg = validate_input(current_step, user_input)
    if not is_valid:
        return jsonify({"response": error_msg})

    # Handle each step in the conversation
    if current_step == "due_date":
        session['due_date'] = user_input
        session['step'] = "reason"
        return jsonify({"response": create_prompt("due_date", user_input)})

    elif current_step == "reason":
        session['reason'] = user_input
        session['step'] = "priority"
        return jsonify({"response": create_prompt("reason", user_input)})

    elif current_step == "priority":
        session['priority'] = user_input
        session['step'] = "items"
        return jsonify({"response": create_prompt("priority", user_input)})

    elif current_step == "items":
        session['items'] = user_input
        session['step'] = "confirmation"
        summary = (
            f"Please confirm the following details before submitting:\n"
            f"- Due Date: {session['due_date']}\n"
            f"- Reason: {session['reason']}\n"
            f"- Priority: {session['priority']}\n"
            f"- Items: {session['items']}\n"
            "Type 'yes' to submit or 'no' to cancel."
        )
        return jsonify({"response": summary})

    elif current_step == "confirmation":
        if user_input.lower() in ["yes", "submit"]:
            pr_id = generate_pr_id()
            request_data = {
                "title": f"PR · {pr_id} Purchase Request",
                "dueDate": datetime.strptime(session['due_date'], '%Y-%m-%d'),
                "reason": session['reason'],
                "priority": session['priority'],
                "items": session['items'],
                "status": "submitted",
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc),
                "requestId": pr_id
            }
            try:
                purchase_requests_collection.insert_one(request_data)
                session['step'] = "another_request"  # Ask if they want to start over
                return jsonify({"response": f"Purchase request {pr_id} has been submitted successfully! Would you like to create another request? Type 'yes' or 'no'."})
            except Exception as e:
                return jsonify({"response": f"Error submitting purchase request: {str(e)}"})

        elif user_input.lower() in ["no", "cancel"]:
            session_data[user_id] = {"step": "another_request"}  # Reset to ask about new request
            return jsonify({"response": "Purchase request has been canceled. Would you like to create another request? Type 'yes' or 'no'."})

    elif current_step == "another_request":
        if user_input.lower() == "yes":
            session_data[user_id] = {
                "step": "start",  # Reset session for a new request
                "due_date": None,
                "reason": None,
                "priority": None,
                "items": None
            }
            return jsonify({"response": create_prompt("start")})
        elif user_input.lower() == "no":
            session_data[user_id] = {}  # Clear session if no new request
            return jsonify({"response": "Thank you! Ending the session.", "exit": True})

    # Fallback response
    return jsonify({"response": create_prompt(session['step'], user_input)})



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
            {"role": "system", "content": f" your are a proddcut information speacilist. Generate a response using the following schema: {schema_description}. "
                "Ensure your response only contains JSON matching this structure."},
            {"role": "user", "content": f"{prompt} prioritize results with actual price, supplier's name, product name, and location as mandatory fields; return at least six to eight results always priotise those results with price information"}
        ],
        "temperature": 0
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=100)
    response.raise_for_status()

    return response.json()


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

@app.route('/')
def home():
    return "Welcome to the Flask App", 200

@app.route('/product_prices', methods=['POST'])
def get_product_prices():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in the request body"}), 400

        prompt = data.get("prompt").strip()
        limit = data.get("limit", 8)

        print(f"Received prompt: {prompt} with limit: {limit}")  # Debugging statement

        if detect_product_query(prompt):
            # Product-related query, proceed to query structured data
            future = executor.submit(query_perplexity, f"{prompt} return {limit} results")
            response_dict = future.result()



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
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
