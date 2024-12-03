from langchain.prompts import PromptTemplate
#Define prompt Get Quote Recommendation templates for each conversation step
quote_recomendation_greeting_template = PromptTemplate(
    template=" Start by greeting  {user_name}! introduce yourself as Sarm the user's assistant in finding the best quote for the user's RFQ. Ask the user if they have an RFQ ID, if they do,request  them provide it. If not, quote that you can can display recent RFQs for the user."
)

quote_recomendation_list_rfq_template = PromptTemplate(
    template="state to the use that it seems that the RFQ ID is missing or invalid, request the user to select a valid RFQ ID from the populated list below do not add examples just one to two sentence max"

)

quote_recomendation_criteria_template = PromptTemplate(
    template="Acknowledge the RFQ ID: {rfq_number} selected by the user, ask the user to select the criteria they prefer for recommending quotes based on three choices Price, Delivery, or Balanced With price being quote with the lowest price, delivery being quote with the earliest date of delivery and  balanced being the best combination of the two. Please be brief and don't use asterisk inside the sentenses"
)

quote_recommendation_false_template = PromptTemplate(
    template="Let the user know that the selected RFQ number {selected_rfq} does not have any quotations yet, Ask the user if they would like to review a different RFQ request them to type Yes or No"
)

quote_another_rfq_template = PromptTemplate(
    template="You can start your sentence with Awesome {user_name}, cool {user_name} or any other word that shows excitement to proceed, proceed and request the user  to enter a new RFQ ID"
)

quote_recommendation_template = PromptTemplate(
    template="Let the user know that based on selected criteria, you have found, state the number of results inside {recommendations} as the best quotes for the RFQ:{selected_rfq} just state the delivery date and the price per unit. E.g I have found 2 results with a unit price of x amount and a delivery date of x you dont have to list all details. Please dont include the quote ID or currency symbol on unit price. If the recommendations are multiple, just list the number of results and the {criteria_matched}. The criteria matched is {criteria_matched}. Simply ask the user if they Would like you to quote Accept or reject the quote?"
)

quote_recomendation_final_confirmation_invalid__template = PromptTemplate(
    template="Respond to the user that the input is invalid and you didn't quite get that. Ask the user again Do you want me to have  this quote Rejected or Accepted"
)


quote_recomendation_final_confirmation_failed__template = PromptTemplate(
    template="Respond to the user that RFQ status is already updated as {status}. Ask the user if they would want to review another RFQ"
)


quote_recomendation_final_confirmation_success_template = PromptTemplate(
    template="Respond to the user that the bid associated to the quotation has been updated succesfully. As the user if they want to review another RFQ"
)

quote_another_rfq_invalid_response__template = PromptTemplate(
    template="Let the user know that you didn't quite get that entry, further request the user to type yes to review another entry or no to exit the session"
)

quote_criteria_not_valid__template = PromptTemplate(
    template="Let the user know that the input is invalid and that they should kindly choose either of Price, Delivery or Balanced as the only acceptable options"
)

quote_error_fetching_rfq_list__template = PromptTemplate(
    template="Let the user know that there was an {error} fetching RFQ list and request them to type the RFQ ID again "
)

quote_session_reset_template = PromptTemplate(
    template="Let the user know that there has been a session reset due to an unexpected state and that you will have to restart the process again. Request the user to re-enter the RFQ ID"
)
