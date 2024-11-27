from langchain.prompts import PromptTemplate

# Define prompt Get Product templates for each conversation step
greeting_template = PromptTemplate(
    input_variables=["user_name"],
    template="Start by greeting  {user_name}! introduce yourself as Sarm the user's assistant in providing product prices and then proceed and simply ask the user to provide the details of the specific product they are looking for?"
)

location_template = PromptTemplate(
    template="Acknowledge the input {product_name}from the user, and Ask the user whats their specified search location? Please don't greet the user again"
)

final_prompt_template = PromptTemplate(
    input_variables=["product_name", "location"],
    template="What is the price of {product_name} in {location}?"
)

option_template = PromptTemplate(
    template="Now that the user has received a couple of results that you sent, respond to the user citing that you hope the results for {product_name} are sufficient. Ask the user if they want to search for another product by typing yes to proceed or no to end the session"
)

exit_template = PromptTemplate(
    template="Say Thank you! to the user and let them know that they can always reach out if they need more assistance"
)

another_product_template = PromptTemplate(
    template="You can start your sentence with Awesome {user_name}, cool {user_name} or any other word that shows excitement to proceed, proceed and request the user  to provide details of the product they are looking for"
)

#
another_product_invalid_response_template = PromptTemplate(
    template="Let the user know that you didn't quite get that, Request the user to type yes to proceed or no to cancel the session and exit"
)
