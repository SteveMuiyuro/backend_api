from langchain.prompts import PromptTemplate

check_request_progress_greeting_template = PromptTemplate(
    template=" Start by greeting  {user_name}! introduce yourself as Sarm the user's assistant in checking the progress of purchase request. Ask the user if they have an Purcharse Request ID, if they do,request them provide it. If not, quote that you can can display recent Purchase Requests that have not been assigned into a workflow for the user.")

check_request_list_of_recent_requests_template = PromptTemplate(
    template="Analyze the {input} and based on the content, you can quote that the can select from a list recent purchase requests. Don't display the list here. request the user to kindly select input the ID of the purchase request you want to assign."
)



check_request_alternative_requests_template = PromptTemplate(
    template="Analyze the {input} and based on the content, Don't display the list here. Request the user to kindly input another ID of the purchase request they want to assign."
)


check_request_selected_template = PromptTemplate(
    template="Confirm from the user that they want to know the current status of purchase request:{input} of title {title}. Request the user to type yes or no"
)


check_request_selected_pr_template = PromptTemplate(
    template="Respond that below is the status of purchase request {title}. Ask the user if they would love to know the status of another Purcharse request by typing yes or no"
)


check_another_check_template = PromptTemplate(
    template="You can start your sentence with Awesome {user_name}, cool {user_name} or any other word that shows excitement to proceed, proceed and request the user  to provide details of the product they are looking for"
)


check_another_check_invalid_template = PromptTemplate(
    template="Analyze the {input} and let the user know that they need to type a yes to proceed with the assignmnet of another request or no to exit the session"
)
