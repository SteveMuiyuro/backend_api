import os
import json
import requests
from flask import Flask, request, jsonify,url_for
from dotenv import load_dotenv
from datetime import datetime, timezone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from flask import Flask, request, jsonify, url_for
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from jsonschema import validate, ValidationError
import re
from flask_caching import Cache
import validators
from flask_cors import CORS
from pymongo import MongoClient
import uuid
import redis


app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv("FLASK_SECRET_KEY")
CORS(app)
load_dotenv()

# Access the API keys
PERPLEXITY_API = os.getenv("PERPLEXITY_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
EBAY_ID = os.getenv("EBAY_ACCESS_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
OPENAI_API_URL ="https://api.openai.com/v1/chat/completions"
CHAT_QUERY_URL = "https://api.perplexity.ai/query"
MONGO_URI = os.getenv("MONGO_URI")
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client.sourcify
purchase_requests_collection = db.requests
workflows_collection = db.workflows
users_collection = db.users
roles_collection = db.roles
request_for_collections = db.requestfors

session_data = {}

# redis_client = redis.Redis(host='localhost', port=6379, db=0) #Local
redis_client = redis.Redis.from_url(os.environ.get('REDIS_URL')) #Production


# Initialize OpenAI's GPT-4 model
llm = ChatOpenAI(model="gpt-4-turbo", api_key=OPEN_AI_KEY)


# Set up memory for conversation
memory = ConversationBufferMemory(return_messages=True,  initial_messages=[
        SystemMessage(content="Please ensure The response for the product step and the location step are one two sentences maximum and always mention the username in your response.")
    ])

# Define prompt templates for each conversation step
greeting_template = PromptTemplate(
    input_variables=["user_name"],
    template="Ask the user the following question, Hello {user_name}, I'm here to assist you in searching for product prices in various locations. What kind of product are you looking for?"
)

location_template = PromptTemplate(
    template="Ask the user the following question, What is your specified search location?"
)

final_prompt_template = PromptTemplate(
    input_variables=["product_name", "location"],
    template="What is the price of {product_name} in {location}?"
)


# Define the conversation chain with memory
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# ThreadPoolExecutor for concurrent API calls
executor = ThreadPoolExecutor(max_workers=10)

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
            {"role": "system", "content": f" your are a product information speacilist. Generate a response using the following schema: {schema_description}. "
                "Ensure your response only contains JSON matching this structure."},
            {"role": "user", "content": f"{prompt} prioritize results with actual price, supplier's name, product name, and location, a legitimate and official website and email as mandatory fields; return at least six to eight results always priotise those results with price information"}
        ],
        "temperature": 0
    }
    response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=100)
    response.raise_for_status()
    return response.json()

def get_product_price_data(prompt, limit):
    try:
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
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=100)
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


def generate_pr_id():
    return f"PR-{uuid.uuid4().hex[:6]}"

# Function to create dynamic prompts based on conversation state
def create_prompt(current_step, user_input=None):
    if current_step == 'start':
        return "Hello! Let's create a purchase request. When is the due date for the request?"
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


# Helper functions for Redis session management
def set_session_data(user_id, data):
    redis_client.setex(f"session:{user_id}", 3600, json.dumps(data))  # Session expires in 1 hour

def get_session_data(user_id):
    session_data = redis_client.get(f"session:{user_id}")
    return json.loads(session_data) if session_data else None

def delete_session_data(user_id):
    redis_client.delete(f"session:{user_id}")


@app.route('/get_product_prices', methods=['POST'])
def get_product_prices():
    user_id = request.json.get('userId')
    user_input = request.json.get('message')
    user_name = request.json.get('userName', 'User')

        # Check if session data exists for the user
    session = get_session_data(user_id)

    # Handle session reset on 'start' command or initialize session if empty
    if user_input.lower() == "start" or not session:
        delete_session_data(user_id)  # Clear previous session data if any
        set_session_data(user_id, {"step": "product"})  # Set initial step to "product"

        # Send the introductory message only once
        response = conversation_chain.predict(input=greeting_template.format(user_name=user_name))
        return jsonify({"response": response})

    # Retrieve the current step from session data
    step = session.get("step")
    print(f"Current step: {step}")  # Debugging

    # Process based on the current step
    if step == "product":
        # User's input is the product name; proceed to the location step
        session["product"] = user_input
        session["step"] = "location"
        set_session_data(user_id, session)

        # Send location template directly to user without model processing
        response = response = conversation_chain.predict(input=location_template.format())
        print(f"Moving to location step with session: {session}")  # Debugging
        return jsonify({"response": response})

    elif step == "location":
        # User's input is the location; proceed to final prompt
        session["location"] = user_input
        product_name = session["product"]
        location = session["location"]

        # Generate the final prompt for retrieving product price data
        prompt = final_prompt_template.format(product_name=product_name, location=location)
        response_data = get_product_price_data(prompt, 8)

        # Ensure the response data is JSON serializable
        response_json = jsonify({
            "response": response_data.json,
            "next_prompt": "Would you like to search for another product? Type yes or no."
        })

        # Update step for another product search
        session["step"] = "another_product"
        set_session_data(user_id, session)

        print(f"Product data retrieved, moving to another_product step with session: {session}")  # Debugging
        return response_json

    elif step == "another_product":
        if user_input.lower() == "yes":
             # Reset session for a new search and keep the user's name
            delete_session_data(user_id)
            set_session_data(user_id, {"step": "product", "user_name": user_name})

            # Send the introductory message again for a new search
            response = conversation_chain.predict(input=greeting_template.format(user_name=user_name))
            print("Restarting session with step 'product'")  # Debugging
            return jsonify({"response": response})
        else:
            delete_session_data(user_id)
            return jsonify({"response": "Thank you! If you need more assistance, feel free to reach out.", "exit": True})

    # If step is unrecognized, reset to initial state
    print("Unexpected step encountered. Resetting session.")  # Debugging
    delete_session_data(user_id)
    set_session_data(user_id, {"step": "product"})

    # Send greeting template directly to user without model processing
    response = greeting_template.format(user_name=user_name)
    return jsonify({"response": "Session reset due to an unexpected state. Let's start over. " + response})


@app.route('/create_request', methods=['POST'])
def create_request():


    # Get userId and message from the request
    user_id = request.json.get('userId')
    user_input = request.json.get('message')

    # # Check if user_id is missing
    # if not user_id:
    #     return jsonify({"response": "User ID is missing. Please try again with a valid user ID."})

    # Initialize session if user_id is new
    if user_input == "start"  or user_id not in session_data:
        session_data[user_id] = {
            "user_id": user_id,
            "step": "due_date",
            "due_date": None,
            "reason": None,
            "priority": None,
            "items": None
        }

    session = session_data[user_id]
    current_step = session['step']

    # If "start" command is received, skip directly to due date prompt
    if user_input == "start" and current_step == "due_date":
        return jsonify({"response": "Hello, When is the due date for the request?"})

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
            f"- Userid: {session['user_id']}\n"
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
                "requestId": pr_id,
                "userid": session["user_id"]
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
                "step": "due_date",  # Reset session and skip directly to due date step
                "due_date": None,
                "reason": None,
                "priority": None,
                "items": None,
                "user_id": user_id
            }
            return jsonify({"response": "When is the due date for the request?"})
        elif user_input.lower() == "no":
            session_data[user_id] = {}  # Clear session if no new request
            return jsonify({"response": "Thank you! Ending the session.", "exit": True})

    # Fallback response if step is not recognized
    return jsonify({"response": "Something went wrong. Please try again."})



#General Querries Logic
@app.route('/product_prices', methods=['POST'])
def product_prices():
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
    app.run(host='0.0.0.0', port=port, debug=True)
