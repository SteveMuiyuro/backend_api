from langchain.prompts import PromptTemplate

create_purchase_order_greeting_template = PromptTemplate(
    template="Start by greeting  {user_name}! introduce yourself as Sarm the user's assistant in creating purchase order for available quotes based on selected RFQ. Request the user to provide the RFQ ID ."
)
create_purchase_order_select_quote_template = PromptTemplate(
    template="Acknowledge the RFQ ID: {rfq_number} and respond that below is the list of quotes available for selection.Do not display the quote details in your response. Request the user to input the quote ID they want to create the PO from"
)

create_purchase_order_no_quote_template = PromptTemplate(
    template="Acknowledge the RFQ ID: {rfq_number} and respond unfortunetly , there are no available quotes for RFQ ID:{rfq_number}. REeust the user to input another RFQ ID"
)

create_purchase_order_quote_selection_invalid_template =  PromptTemplate(
    template="respond unfortunetly the quote ID:{input}, is invalid, request the user to enter the correct quote id"
)

create_purchase_order_quote_confirmation_template = PromptTemplate(
    template="Request the user to review the details of the quote ID:{input} below. Do not display the details of the quote in your response. Request the user to confirm wheather they want to proceed to create PO by typing yes or make changes by typing no"
)

create_purchase_order_po_submitted_template = PromptTemplate(
    template="Inform the user that the po for quote id:{input} has been submitted succesfully. Ask the user if they would like to create another PO by typing yes or no to cancel"
)

create_purchase_order_po_cancelled_template = PromptTemplate(
    template="Respond that the PO for Quote ID {input} has been cancelled. Thank the user for their time and close the session")

check_purchase_order_invalid_template = PromptTemplate(
    template="Analyze the {input} and let the user know that they need to type a yes to proceed with PO creation or no to exit the session")

create_another_purchase_order_template = PromptTemplate(
    template="You can start your sentence with Awesome {user_name}, cool {user_name} or any other word that shows excitement to proceed, proceed and request the user  to enter a new RFQ ID")

create_another_purchase_order_invalid_response_template = PromptTemplate(
    template="Let the user know that you didn't quite get that entry, further request the user to type yes to review another entry or no to exit the session"
)
