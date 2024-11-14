import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from jsonschema import  ValidationError
from flask_cors import CORS
from helpers.product_prices_helpers import  detect_product_query, get_product_price_data, query_general
from helpers.get_product_prices_helpers import get_session_data, set_session_data, delete_session_data

app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv("FLASK_SECRET_KEY")
CORS(app)
load_dotenv()

# Access the API keys
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI's GPT-4 model
llm = ChatOpenAI(model="gpt-4-turbo", api_key=OPEN_AI_KEY)

# Set up memory for conversation
memory = ConversationBufferMemory(return_messages=True,  initial_messages=[
        SystemMessage(content="You are a friendly assistant helping the user get prices for the various products based on location. Please ensure The response for the product step and the location step are one two sentences maximum and always mention the user's name in your response.")
    ])

# Define prompt templates for each conversation step
greeting_template = PromptTemplate(
    input_variables=["user_name"],
    template="Start by greeting the user, Hello! {user_name},  and then proceed and ask the user to provide the details of the specific product they are looking for?"
)

location_template = PromptTemplate(
    template="Acknowledge the response from previous prompt that contains {product_name} if it exits. Ask the user whats their specified search location?"
)

final_prompt_template = PromptTemplate(
    input_variables=["product_name", "location"],
    template="What is the price of {product_name} in {location}?"
)

option_template = PromptTemplate(
    template="Now that the user has received a couple of results that you sent, respond to the user citing that you hope the results for {product_name} are sufficient. Ask the user if they want to search for another product by typing yes to proceed or no to end the session"
)

exit_template = PromptTemplate(
    template="Say Thank you! to the user and let them know that they can always reach out if they ned more assistance"
)

another_product_template = PromptTemplate(
    template="You can start your sentence with Awesome {user_name}, cool {user_name} or any other word that shows excitement to proceed, proceed and request the user  to provide details of the product they are looking for"
)


# Define the conversation chain with memory
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# ThreadPoolExecutor for concurrent API calls
executor = ThreadPoolExecutor(max_workers=10)


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
