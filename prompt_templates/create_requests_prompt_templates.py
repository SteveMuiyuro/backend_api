from langchain.prompts import PromptTemplate

create_request_greetings_template_template = PromptTemplate(
    template="start by greeting the user {user_name} introduce yourself as Sarm and let the user know that you are their assistant in creating a purchase request. Proceed and ask for the request's due date in the format YYYY-MM-DD and ensure the user inputs today's date or later date"
)

create_request_reason_template_template = PromptTemplate(
    template="Respond that the due date:{input} has been noted. Request the user to provide the reason for creating the purchase request"
)

create_request_priority_template_template = PromptTemplate(
    template="Respond that that the reason:{input} has been noted. simply request the user to provide the priority for the purchase based on the options below: Don't mention the options"
)

create_request_recored_items_template_template = PromptTemplate(
    template="Respond that the priority: {input} has been noted. Based on the reason provided:{reason}, Ask the user what items they are looking to purchase in the format 10 Laptops or 10 Chairs")


create_request_invalid_date_template_template = PromptTemplate(
    template="Respond that the due date cannot be in the past. Request them to  provide a valid date (today or later). "
)

create_request_invalid_format_date_template_template = PromptTemplate(
    template="Respond that the user that you can't save the date in that format. Suggest that the due date should be in the format YYYY-MM-DD. Dont acknowledge the response but you can use the word sorry or a synonym of the word"
)

create_request_invalid_priority_template_template = PromptTemplate(
    template="Respond that the priority must be one of the following options(Urgent, High, Medium, or Low)."
)

create_request_summary_template = PromptTemplate(
    template="Respond that the items: {items} has been noted. Request the user to confirm the following details before submitting\n:Due Date:{due_date}\n Reason:{reason}\n Priority:{priority} Items:{items}. Ask the user if they would want to submit the reuest by typing yes or cancel by typing no."
)

create_request_another_template = PromptTemplate(
    template="Let the user know that the purchase request PR-{input} has been submitted successfully! and ask then if they Would like to create another request? by typing 'yes' or 'no'. "
)

create_request_cancel_template = PromptTemplate(
    template="Respond that the purchase request PR-{input} has been cancelled! and ask them if they Would like to create another request? by typing 'yes' or 'no'. "
)

create_request_session_reset_template = PromptTemplate(
    template="Respond that there has been a session reset due to an unxpected state and that you will have to restart the process again. Request the user to re-enter the due date for the request"
)

create_request_confirmation_invalid_template = PromptTemplate(
    template="Respond that know that you didn't quite get that entry, further request the user to type yes to submit the request or no to cancel the request"
)


create_another_request_template = PromptTemplate(
    template="You can start your sentence with Awesome {user_name}, cool {user_name} or any other word that shows excitement to proceed, proceed and request the user  to enter the due date for the new request"
)

create_request_another_confirmation_invalid_template = PromptTemplate(
    template="Respond that you didn't quite get that entry, further request the user to type yes to create another request or no to exit the session"
)

create_request_invalid_priority_template = PromptTemplate(
    template="Respond that you {input} is not a valid priority, request the user to typewother of the follwing priorities: Urgemt, High, Medium or Low"
)
