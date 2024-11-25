from langchain.prompts import PromptTemplate
#Define prompt Get Quote Recommendation templates for each conversation step
quote_recomendation_greeting_template = PromptTemplate(
    input_variables=["user_name"],
    template=" Start by greeting  {user_name}! introduce yourself as Sarm the user's assistant in finding the best quote for the user's RFQ. Ask the user if they have an RFQ ID, if they do,request  them provide it. If not, quote that you can can display recent RFQs for the user."
)

quote_recomendation_list_rfq_template = PromptTemplate(
    template="state to the use that it seems that the RFQ ID is missing or invalid, request the user to select a valid RFQ ID from the populated list below do not add examples just one to two sentence max"

)

quote_recomendation_criteria_template = PromptTemplate(
    template="Acknowledge the RFQ ID: {rfq_number} selected by the user, ask the user to select the criteria they prefer for recommending quotes based on three choices Price, Quantity, or Balanced With price being quote with the lowest price,quantity being quote with the highest quantity and  balanced being the best combination of the two. Please be brief and don't use asterisk inside the sentenses"
)

quote_recommendation_false_template = PromptTemplate(
    template="Let the user know that the selected RFQ number {selected_rfq} does not have any quotations yet, Ask the user if they would like to review a different RFQ request them to type Yes  or No"
)

quote_another_rfq_template = PromptTemplate(
    template="You can start your sentence with Awesome {user_name}, cool {user_name} or any other word that shows excitement to proceed, proceed and request the user  to enter a new RFQ ID"
)

quote_exit_template = PromptTemplate(
    template="Say Thank you! to the user and let them know that they can always reach out if they need more assistance"
)

quote_recommendation_template = PromptTemplate(
    input_variables=["recommendations"],
    template="Let the user know that based on selected criteria, the best quotes for the RFQ:{selected_rfq} as listed below is are is for:{recommendations} Please dont include the quote ID or currency symbol on unit price. The criteria matched is : {criteria_matched} simply ask the user if they Would like you to have the RFQ associated to this quote Accepted or rejected?"
)

quote_recomendation_final_confirmation_invalid__template = PromptTemplate(
    template="Respond to the user that the input is invalid and you didn't quite get that. Ask the user again Do you want me to have the bid associated to this quote Rejected or Accepted"
)


quote_recomendation_final_confirmation_failed__template = PromptTemplate(
    template="Respond to the user that RFQ status is already updated as {status}. ASk the user if they would want to review another RFQ by typing yes or no"
)


quote_recomendation_final_confirmation_success_template = PromptTemplate(
    template="Respond to the user that the RFQ associated to the quotation has been updated succesfully. As the user if they want to review another quote by typing yes or no"
)

quote_another_rfq_invalid_response__template = PromptTemplate(
    template="Let the user know that you didn't quote get that entry, further request the user to type yes to review another entry or no to exit the session"
)

quote_criteria_not_valid__template = PromptTemplate(
    template="Let the user know that the input is invalid and that they should kindly choose either of Price, Quantity or Balanced as the only acceptable options"
)

quote_error_fetching_rfq_list__template = PromptTemplate(
    template="Let the user know that there was an {error} fetching RFQ list and request them to type the RFQ ID again "
)
