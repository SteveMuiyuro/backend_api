import os
import json
from flask import Flask, request, jsonify
from bson import ObjectId
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from jsonschema import  ValidationError
from flask_cors import CORS
import ast
from pymongo import MongoClient
from helpers.product_prices_helpers import  detect_product_query, get_product_price_data, query_general
from helpers.get_product_prices_helpers import get_session_data, set_session_data, delete_session_data
from prompt_templates.get_product_price_prompt_templates import greeting_template, location_template,exit_template,another_product_template,option_template, final_prompt_template
from prompt_templates.quote_recomendation_templates import quote_recomendation_greeting_template, quote_recomendation_criteria_template,quote_recomendation_final_confirmation_template,quote_recomendation_list_rfq_template,quote_recommendation_template
app = Flask(__name__, static_folder='static')
CORS(app)
load_dotenv()

# Access the API keys
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URL = os.getenv("MONGO_URI")

# MongoDB setup
client = MongoClient(MONGO_URL)
db = client.sourcify
bids_collection = db.bids
requests_fors = db.requestfors
results_bids = bids_collection.find({})
request_id = "6710048ca489320770e1ce3e"



def find_request_id(input_value, db):
    try:
        # Try converting the input to an integer
        try:
            rfq_input = int(input_value)
        except ValueError:
            raise ValueError("Input must be a numeric value.")

        # Check if the requestId exists as a number
        rfq = db.requestfors.find_one({"requestId": rfq_input}, {"requestId": 1, "_id": 0})

        # If not found, check if it's stored as a string
        if not rfq:
            rfq = db.requestfors.find_one({"requestId": str(rfq_input)}, {"requestId": 1, "_id": 0})

        # If a match is found, return the result
        if rfq:
            print(f"RFQ found: {rfq}")
            return rfq
        else:
            print(f"No RFQ found for requestId: {rfq_input}")
            return None

    except Exception as e:
        print(f"Error while querying RFQs: {str(e)}")
        return None


def getBids(requestForID, db):
    if isinstance(requestForID, str):
        try:
            requestForID = ast.literal_eval(requestForID)
        except (ValueError, SyntaxError):
            print(f"Invalid format for requestForID: {requestForID}")
            return []

    if isinstance(requestForID, dict) and 'requestId' in requestForID:
        requestForID = requestForID['requestId']

    try:
        newRequestId = int(requestForID)
    except ValueError:
        print(f"Invalid requestForID format: {requestForID}")
        return []

    request_document = db.requestfors
    results = request_document.find({"requestId": newRequestId}, {"bids": 1, "_id": 0})

    # Flatten the bids list and convert ObjectId to string
    bids = []
    for result in results:
        if "bids" in result:
            # Flatten nested arrays and convert ObjectId to string
            bid = result["bids"]
            if isinstance(bid, list):
                for bid_item in bid:
                    # Convert ObjectId to string if it's present
                    if isinstance(bid_item, ObjectId):
                        bids.append(str(bid_item))
                    else:
                        bids.append(bid_item)

    print(f"These are the combined bids: {bids}")
    return bids

def getQuotationsForBids(bidIds, db):
    # Initialize an empty list to store the quotations for each bid
    all_quotations = []

    for bidId in bidIds:
        # Ensure the bidId is a valid ObjectId (if it's a string, convert it)
        if isinstance(bidId, str):
            try:
                bidId = ObjectId(bidId)  # Convert to ObjectId from string
            except Exception as e:
                print(f"Invalid ObjectId format: {bidId} - {e}")
                all_quotations.append("Invalid ObjectId format")
                continue

        bids_collection = db.bids

        # Query the bids collection to find the quotations for the given bidId
        result = bids_collection.find_one({"_id": bidId}, {"quotations": 1, "_id": 0})

        if result and 'quotations' in result:
            # If quotations exist for the bid, add them as a list of strings
            quotation_ids = [str(quotation_id) for quotation_id in result['quotations']]
            all_quotations.extend(quotation_ids)  # Add the quotations to the result list
        else:
            # If no quotations are found, add an empty list
            all_quotations.extend([])

    return all_quotations

def getPurchaseID(requestForID, db):
    request_document = db.requestfors
    result = request_document.find_one(
        {"_id": ObjectId(requestForID)},
        {"purchaseRequest": 1, "requestId": 1, "_id": 0}
    )
    if result:
        purchase_request = result.get("purchaseRequest")
        request_id = result.get("requestId")

        return {"purchaseRequest": purchase_request, "requestId": request_id}
    else:
        return None

def getPurchaseRequestTitle(purcharseRequestID, db):
    request_collection = db.requests
    result = request_collection.find_one({"_id":ObjectId(purcharseRequestID)}, {"title":1, "_id":0})
    if result:
        title = str(result["title"])
        return title
    else:
        return None


def listAllRFQs(db):
    bids_collection = db.bids
    bid_lists = bids_collection.find({}, {"request": 1, "_id": 0})

    # Build the final list with requestId and title
    final = []
    for doc in bid_lists:
        purchase_request = doc["request"]
        # Get the purchase request ID and requestId
        purchase_data = getPurchaseID(purchase_request, db)
        request_id = purchase_data["requestId"]
        purchase_request_id = purchase_data["purchaseRequest"]

        # Get the title for the purchase request
        title = getPurchaseRequestTitle(purchase_request_id, db)

        # Format the final output
        final.append(f"RFQ ID:{request_id} Title:{title}")

    return final


# Initialize OpenAI's GPT-4 model
llm = ChatOpenAI(model="gpt-4-turbo", api_key=OPEN_AI_KEY)

# Set up memory for conversation
memory = ConversationBufferMemory(return_messages=True,  initial_messages=[
        SystemMessage(content="You are a friendly assistant helping the user on various issues from getting prices for various products to creating requesrs and RFQ assignments.")
    ])

# Define the conversation chain with memory
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# ThreadPoolExecutor for concurrent API calls
executor = ThreadPoolExecutor(max_workers=10)


# Route for recommending best quotes
@app.route('/recommend_quotes', methods=['POST'])
def recommend_best_quotes():
    user_id = request.json.get('userId')
    user_input = request.json.get('message')
    user_name = request.json.get('userName')
    set_session_data(user_id, {"step": "rfq_checking"})
    session = get_session_data(user_id)
    delete_session_data(None)


    # Handle session initialization or reset
    # Reset session
    if user_input == "start" or not session:
        response = conversation_chain.predict(input=quote_recomendation_greeting_template.format(user_name=user_name))
        return jsonify({"response": response})

    step = session.get("step", "rfq_checking")  # Default to "rfq_checking" if step is missing
    print(f"User {user_id} is at step: {step}")
    if step not in ["rfq_checking", "criteria_selection", "quote_recommendation", "quote_evaluation", "final_confirmation"]:
        print(f"Unexpected step '{step}' for user {user_id}. Resetting session.")
        delete_session_data(user_id)
        set_session_data(user_id, {"step": "rfq_checking"})
        return jsonify({"response": "Your session has been reset due to an invalid state. Please start again."})


    if step == "rfq_checking":
        rfq_input = user_input.strip()
        rfq = find_request_id(rfq_input, db)
        bid = getBids(str(rfq), db)
        print(getQuotationsForBids(bid, db))
        if not rfq:
            try:
                # Fetch all available RFQ IDs if the provided ID is not found

                rfq_ids = listAllRFQs(db)

                # Add the RFQs to the session
                session["rfqs"] = rfq_ids
                set_session_data(user_id, session)

                # Respond with the list of available RFQ IDs
                response = {
                    "response": conversation_chain.predict(input=quote_recomendation_list_rfq_template.format()),
                    "available_rfqs": rfq_ids,
                }
                return jsonify(response)
            except Exception as e:
                return jsonify({"response": f"Error fetching RFQ list: {str(e)}"})


        # If match is found, proceed
        session["selected_rfq"] = str(rfq)  # Store as string for JSON serialization
        session["step"] = "criteria_selection"
        set_session_data(user_id, session)

        response = conversation_chain.predict(input=quote_recomendation_criteria_template.format(rfq_number = rfq))
        return jsonify({"response": response})

    elif step == "criteria_selection":
        rfq_selection = user_input.strip()

        # Validate that "rfqs" exists in the session
        if "rfqs" not in session or not session["rfqs"]:
            return jsonify({"response": "RFQs list is not available in the session. Please restart the process."})

        # Validate the selection is a valid index
        try:
            rfq_index = int(rfq_selection) - 1
            if rfq_index < 0 or rfq_index >= len(session["rfqs"]):
                return jsonify({"response": "Invalid RFQ selection. Please choose a valid option from the list."})
        except ValueError:
            return jsonify({"response": "Invalid input. Please enter a valid number corresponding to an RFQ."})

        selected_rfq = session["rfqs"][rfq_index]
        session["selected_rfq"] = selected_rfq
        session["step"] = "quote_recommendation"
        set_session_data(user_id, session)

        response = conversation_chain.predict(input=quote_recomendation_criteria_template.format())
        return jsonify({"response": response})


    elif step == "quote_evaluation":
    # Assume user has selected a quote from the displayed options
        selected_quote_id = user_input  # Use input to map to a quote ID
        selected_quote = bids_collection.find_one({"_id": selected_quote_id})

        if selected_quote:
            session["selected_quote"] = selected_quote
            session["step"] = "final_confirmation"
            set_session_data(user_id, session)

            # Generate final confirmation prompt
            quote_details = f"Vendor: {selected_quote['vendor']}, Total Price: {selected_quote['total_price']}, Delivery Date: {selected_quote['delivery_date']}"
            response = conversation_chain.predict(
                input=quote_recomendation_final_confirmation_template.format(quote_details=quote_details)
            )
            return jsonify({"response": response})
        else:
            response = "The selected quote could not be found. Please try again."
            return jsonify({"response": response})

    elif step == "quote_recommendation":
        # Evaluate and recommend quotes based on criteria
        criteria = user_input.lower()
        selected_rfq = session["selected_rfq"]
        rfq_id = selected_rfq["id"]

        # Fetch and evaluate quotes from the database
        quotes = list(bids_collection.find({"rfq_id": rfq_id}))
        if not quotes:
            return jsonify({"response": "No quotes available for the selected RFQ."})

        # Apply filtering logic based on criteria
        # For demonstration purposes, we'll sort quotes by price (extendable with other criteria)
        if "price" in criteria:
            quotes = sorted(quotes, key=lambda q: q["total_price"])

        recommendations = "\n".join([
            f"Vendor: {quote['vendor']} - Total Price: {quote['total_price']}, Delivery: {quote['delivery_date']} (Criteria matched: Price)"
            for quote in quotes
        ])

        session["step"] = "final_confirmation"
        set_session_data(user_id, session)

        response = conversation_chain.predict(input=quote_recommendation_template.format(recommendations=recommendations))
        return jsonify({"response": response})

    elif step == "final_confirmation":
        # Handle user actions: accept, reject, message
        action = user_input.lower()
        if action == "accept":
            # Simulate acceptance logic
            session["step"] = "exit"
            set_session_data(user_id, session)
            response = "The quote has been accepted and marked as selected."
        elif action == "reject":
            response = "The quote has been rejected. Would you like to view other quotes?"
        elif action == "message":
            response = "Message functionality is under development. Please wait for updates."
        else:
            response = "Invalid input. Please type 'accept', 'reject', or 'message'."
        return jsonify({"response": response})

    elif step == "exit":
        # End the session
        delete_session_data(user_id)
        response = conversation_chain.predict(input=exit_template.format())
        return jsonify({"response": response, "exit": True})

    # Unhandled step
    delete_session_data(user_id)
    set_session_data(user_id, {"step": "rfq_selection"})
    return jsonify({"response": "Session reset due to an unexpected state. Please start again."})


@app.route('/get_product_prices', methods=['POST'])
def get_product_prices():
    user_id = request.json.get('userId')
    user_input = request.json.get('message')
    user_name = request.json.get('userName')

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
        response = conversation_chain.predict(input=location_template.format(product_name=session["product"]))
        return jsonify({"response": response})

    elif step == "location":
        # User's input is the location; proceed to final prompt
        session["location"] = user_input
        product_name = session["product"]
        location = session["location"]

        # Generate the final prompt for retrieving product price data
        prompt = final_prompt_template.format(product_name=product_name, location=location)
        response_data = get_product_price_data(prompt, 8)
        response = conversation_chain.predict(input=option_template.format(product_name=product_name))

        # Ensure the response data is JSON serializable
        response_json = jsonify({
            "response": response_data.json,
            "next_prompt": response
        })

        # Update step for another product search
        session["step"] = "another_product"
        set_session_data(user_id, session)
        return response_json

    elif step == "another_product":
        if user_input.lower() == "yes":
             # Reset session for a new search and keep the user's name
            delete_session_data(user_id)
            set_session_data(user_id, {"step": "product"})

            # Send the introductory message again for a new search
            response = conversation_chain.predict(input=another_product_template.format(user_name=user_name))
            return jsonify({"response": response})
        else:
            delete_session_data(user_id)
            response = conversation_chain.predict(input=exit_template.format())
            return jsonify({"response": response, "exit": True})

    # If step is unrecognized, reset to initial state
    delete_session_data(user_id)
    set_session_data(user_id, {"step": "product"})

    # Send greeting template directly to user without model processing
    conversation_chain.predict(input=greeting_template.format(user_name=user_name))
    return jsonify({"response": "Session reset due to an unexpected state. Let's start over. " + response})



#General Querries Logic
@app.route('/product_prices', methods=['POST'])
def product_prices():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in the request body"}), 400

        prompt = data.get("prompt").strip()
        limit = data.get("limit", 8)

        if detect_product_query(prompt):
            return get_product_price_data(prompt, limit)

        else:
            try:
                future = executor.submit(query_general, prompt)
                response_dict = future.result()

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
