from langchain.prompts import PromptTemplate

create_request_greetings_template_template = PromptTemplate(
    template="start by greeting the user {user_name} introduce yourself as Sarm and let the user know that you are their assistant in creating a purchase request. Proceed and ask for the request's due date"
)

create_request_reason_template_template = PromptTemplate(
    template="Let the user know tha the due date:{input} has been noted. Request the user to provide the reason for the purchase"
)

create_request_priority_template_template = PromptTemplate(
    template="Let the user know that the reason:{input} has been noted. Request the user to provide the priority for the purchase based on the following options (Urgent, High, Medium, Low)"
)

create_request_recored_items_template_template = PromptTemplate(
    template="Let the user know that the priority: {input} has been noted. Ask the user what items they are looking to purchase")


create_request_another_template_template = PromptTemplate(
    template="Ask the user if they would love to create another request"
)

create_request_invalid_date_template_template = PromptTemplate(
    template="Let the user know that the due date cannot be in the past. Request them to  provide a valid date (today or later). "
)

create_request_invalid_format_date_template_template = PromptTemplate(
    template="Request the user to enter the due date in YYYY-MM-DD format."
)

create_request_invalid_priority_template_template = PromptTemplate(
    template="Let the user know that priority must be one of the following options(Urgent, High, Medium, or Low)."
)


create_request_summary_template = PromptTemplate(
    template="Let the user know that the items: {items} has been noted. Request the user to confirm the following details before submitting\n:Due Date:{due_date}\n Reason:{reason}\n Priority:{priority} Items:{items}. Ask the user if they would want to submit the reuest by typing yes or cancel by typing no."
)

create_request_another_template = PromptTemplate(
    template="Let the user know that the purchase request PR-{input} has been submitted successfully! and ask then if they Would like to create another request? by typing 'yes' or 'no'. "
)

create_request_cancel_template = PromptTemplate(
    template="Let the user know that the purchase request PR-{input} has been cancelled! and ask them if they Would like to create another request? by typing 'yes' or 'no'. "
)


create_request_session_reset_template = PromptTemplate(
    template="Let the user know that there has been a session reset due to an unxpected state and that you will have to restart the process again. Request the user to re-enter the due date for the request"
)
