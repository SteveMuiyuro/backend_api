import os
import json
from flask import Flask, request, jsonify
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
from pymongo import MongoClient
from helpers.product_prices_helpers import  detect_product_query, get_product_price_data, query_general
from helpers.get_product_prices_helpers import get_session_data, set_session_data, delete_session_data
from helpers.recommend_qoutes_helpers import evaluate_quotes, find_request_id, getBids, getQuotationDetails, getQuotationsForBids, listAllRFQs, updateBidStatus
from prompt_templates.get_product_price_prompt_templates import greeting_template, location_template,exit_template,another_product_template,option_template, final_prompt_template
from prompt_templates.quote_recomendation_templates import quote_recomendation_greeting_template, quote_recomendation_criteria_template,quote_recomendation_list_rfq_template,quote_recommendation_template, quote_recommendation_false_template,quote_another_rfq_template,quote_exit_template,quote_recomendation_final_confirmation_failed__template,quote_recomendation_final_confirmation_invalid__template,quote_recomendation_final_confirmation_success_template,quote_another_rfq_invalid_response__template,quote_criteria_not_valid__template, quote_error_fetching_rfq_list__template
app = Flask(__name__, static_folder='static')
CORS(app)
load_dotenv()

# Access the API keys
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URL = os.getenv("MONGO_URI")

# MongoDB setup
client = MongoClient(MONGO_URL)
db = client.sourcify

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

    session = get_session_data(user_id)
    # Reset session
    if user_input == "start":
        # Reset session only on "start" and set a flag
        set_session_data(user_id, {"step": "rfq_checking", "greeting_sent": True})
        response = conversation_chain.predict(input=quote_recomendation_greeting_template.format(user_name=user_name))
        return jsonify({"response": response, "status": False})

    # Check if the session exists and greeting has already been sent
    if not session:
        # If no session, initialize with greeting already sent
        set_session_data(user_id, {"step": "rfq_checking", "greeting_sent": True})
        response = conversation_chain.predict(input=quote_recomendation_greeting_template.format(user_name=user_name))
        return jsonify({"response": response, "status": False})
    elif not session.get("greeting_sent"):
        # If session exists but greeting was not sent, send it
        session["greeting_sent"] = True
        set_session_data(user_id, session)
        response = conversation_chain.predict(input=quote_recomendation_greeting_template.format(user_name=user_name))
        return jsonify({"response": response, "status": False})



    step = session.get("step")  # Default to "rfq_checking" if step is missing
    print(f"User {user_id} is at step: {step}")
    if step not in ["rfq_checking", "criteria_selection", "quote_recommendation", "quote_evaluation", "final_confirmation", "another_rfq"]:
        print(f"Unexpected step '{step}' for user {user_id}. Logging and resetting session.")
        # Log additional details for debugging
        print(f"Session data: {session}")
        delete_session_data(user_id)
        set_session_data(user_id, {"step": "rfq_checking"})
        return jsonify({"response": "Your session has been reset due to an invalid state. Please start again."})


    if step == "rfq_checking":
        rfq_input = user_input.strip()
        rfq = find_request_id(rfq_input, db)
        if not rfq:
            try:
                rfq_ids = listAllRFQs(db)
                session["rfqs"] = rfq_ids
                set_session_data(user_id, session)
                response = {
                    "response": conversation_chain.predict(input=quote_recomendation_list_rfq_template.format()),
                    "available_rfqs": rfq_ids,
                }
                return jsonify(response)
            except Exception as e:
                session["step"] = "rfq_checking"
                response = conversation_chain.predict(input=quote_error_fetching_rfq_list__template.format(error=str(e))),
                return jsonify({"response":response})


        # If match is found, proceed
        session["selected_rfq"] = rfq  # Store as string for JSON serialization
        session["selected_bids"] = getBids(rfq, db)
        session["selected_quotes"] = getQuotationsForBids(session["selected_bids"], db)
        session["step"] = "criteria_selection"
        set_session_data(user_id, session)

        response = conversation_chain.predict(input=quote_recomendation_criteria_template.format(rfq_number = rfq))
        return jsonify({"response": response})

    elif step == "criteria_selection":
        criteria = user_input.strip().capitalize()
        print(f"User selected criteria: {criteria}")

        valid_criteria = ["Price", "Quantity", "Balanced"]

        if criteria not in valid_criteria:
            response= conversation_chain.predict(input=quote_criteria_not_valid__template.format())
            return jsonify({"response": response})

        quote_details = getQuotationDetails(session["selected_quotes"], db)
        if not quote_details:
            selected_rfq = session["selected_rfq"]
            session["step"] = "another_rfq"
            set_session_data(user_id, session)
            response = conversation_chain.predict(
            input=quote_recommendation_false_template.format(selected_rfq = selected_rfq))
            return jsonify({"response": response})

        recommended_quote = evaluate_quotes(quote_details, criteria, weights=(0.4,0.6))
        session["selected_quote_id"] = recommended_quote[0]['quote']['id']

        if criteria == "Price":
            criteria_match = "Lowest Price"
        elif criteria == "Quantity":
            criteria_match="Highest Quantity"
        else:
            criteria_match="Balance between Lowest Price and Hightest Quantity"

        # Save the valid criteria and move to the next step
        session["selected_criteria"] = criteria
        session["step"] = "final_confirmation"
        set_session_data(user_id, session)

        response = conversation_chain.predict(
            input=quote_recommendation_template.format(recommendations=recommended_quote, criteria_matched=criteria_match)
        )
        return jsonify({"response": response, "best_quotes":recommended_quote})


    elif step == "final_confirmation":
        # Handle user actions: accept, reject, success
        action = user_input.lower()
        selected_quote = session.get("selected_quote_id")

        # Validate session state before proceeding
        if not selected_quote:
            print(f"Missing selected_quote_id in session for user {user_id}. Resetting session.")
            delete_session_data(user_id)
            set_session_data(user_id, {"step": "rfq_checking"})
            return jsonify({"response": "Session reset due to an unexpected state. Please start again."})

        context = updateBidStatus(selected_quote, action, db)
        if context == "Invalid":
            session["step"] = "final_confirmation"
            set_session_data(user_id, session)
            response = conversation_chain.predict(input=quote_recomendation_final_confirmation_invalid__template.format())
            return jsonify({"response": response})

        if context == "Failed":
            session["step"] = "another_rfq"
            set_session_data(user_id, session)
            response = conversation_chain.predict(input=quote_recomendation_final_confirmation_failed__template.format())
            return jsonify({"response": response})

        if context == "Success":
            session["step"] = "another_rfq"
            set_session_data(user_id, session)
            response = conversation_chain.predict(input=quote_recomendation_final_confirmation_success_template.format())
            return jsonify({"response": response})

            # Add a new branch in the main handler for "another_rfq"
    elif step == "another_rfq":
        action = user_input.lower()
        if action == "yes":
            greeting_sent = session.get("greeting_sent", True)  # Default to True for safety
            delete_session_data(user_id)
            set_session_data(user_id, {"step": "rfq_checking", "greeting_sent": greeting_sent})
            response = conversation_chain.predict(
                input=quote_another_rfq_template.format(user_name=user_name)
            )
            return jsonify({"response": response})
        elif action == "no":
            delete_session_data(user_id)
            response = conversation_chain.predict(
                input=quote_exit_template.format()
            )
            return jsonify({"response": response, "exit": True})
        else:
            # Handle invalid inputs for the yes/no decision
            response = conversation_chain.predict(
                input=quote_another_rfq_invalid_response__template.format()
            )
            return jsonify({"response": response})


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
