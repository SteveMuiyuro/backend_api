from langchain.prompts import PromptTemplate

create_rfq_greetings_template_template = PromptTemplate(
    template="start by greeting the user {user_name} introduce yourself as Sarm and let the user know that you are their assistant in creating an RFQ(Request for Quotation). Proceed and ask the user to provide the purchase request ID. If not, quote that you can can display recent purchase Requests that do not have an RFQ assigned to them"
)

create_rfq_list_of_recent_requests_template = PromptTemplate(
    template="Analyze the {input} and based on the content, you can quote that the can select from a list recent purchase requests. Don't display the list here. request the user to kindly select input the ID of the purchase request you want to assign."
)


create_rfq_due_date_template = PromptTemplate(
    template="Akcnowledge that {input} has been noted. Ask for the submission deadline for quotes in the format YYYY-MM-DD and ensure the user inputs today's date or later date"
)


create_rfq_description_template = PromptTemplate(
    template="Akcnowledge that {input} has been noted. Request that the user provide the description of the RFQ"
)

create_rfq_confirmation_template = PromptTemplate(
    template="Akcnowledge that {input}  and  request the user to review the  RFQ details below (don't list or display the details here). Request the user to confirm wheather they want to proceed with submitting the request by typing 'yes' or 'no'. "
)

create_rfq_another_template = PromptTemplate(
    template="Let the user know that the RFQ-{input} has been submitted successfully! and ask them if they Would like to create another RFQ request? by typing 'yes' or 'no'. "
)

create_another_rfq_template = PromptTemplate(
    template="You can start your sentence with Awesome {user_name}, cool {user_name} or any other word that shows excitement to proceed, proceed and request the user to enter a new request ID"
)

create_rfq_revise_details_template = PromptTemplate(
    template="The user seem to have made a mistake while providing data, Respond that you need to restart the details collection for clarity. Requset the user to enter the due date in the format YYYY-MM-DD and ensure the user inputs today's date or later date"
)
