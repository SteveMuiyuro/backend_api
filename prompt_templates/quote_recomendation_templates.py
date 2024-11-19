from langchain.prompts import PromptTemplate
#Define prompt Get Quote Recommendation templates for each conversation step
quote_recomendation_greeting_template = PromptTemplate(
    input_variables=["user_name"],
    template=" Start by greeting  {user_name}! introduce yourself as Sarm the user's assistant in finding the best quote for the user's RFQ. Ask the user if they have an RFQ ID or Title, if they do,request  them provide it. If not, quote that you can can display recent RFQs for the user."
)

quote_recomendation_list_rfq_template = PromptTemplate(
    template="state to the use that  it seems they don't have an ID or the RFQ ID  entered has an invalid format, request the user to select a valid RFQ ID from the populated list below do not add examples just one to two sentence max"

)

quote_recomendation_criteria_template = PromptTemplate(
    template="Acknowledge the {rfq_number} selected by the user, ask the user to select the criteria they prefer for recommending quotes based on three choices Price, Quantity, or Delivery Date. Give the user the option to combine criteria  e.g., 'Price and Delivery Date'."
)

quote_recommendation_template = PromptTemplate(
    input_variables=["recommendations"],
    template="Let the user know that based on selected criteria, the best quotes are:\n{recommendations}\n Ask the user if they Would you like to accept, reject, or message the vendor for more information?"
)

quote_recomendation_final_confirmation_template = PromptTemplate(
    template="Acknowledge the selection by quoting Thank you! and letting the user know that the selection has been noted. Ask the user if they Would you like to monitor the progress of the request or exit the session?"
)
