from langchain.prompts import PromptTemplate

check_request_progress_greeting_template = PromptTemplate(
    template=" Start by greeting  {user_name}! introduce yourself as Sarm the user's assistant in checking the progress of purchase request. Ask the user if they have an Purcharse Request ID, if they do,request them provide it. If not, quote that you can can display recent Purchase Requests that have not been assigned into a workflow for the user.")

check_request_list_of_recent_requests_template = PromptTemplate(
    template="Analyze the {input} and based on the content, you can quote that the can select from a list recent purchase requests. Don't display the list here. request the user to kindly select input the ID of the purchase request you want to assign."
)

check_request_selected_template = PromptTemplate(
    template="Confirm from the user that they want to know the current status of purchase request:{input} of titled {title}. Request teh user to type yes or no"
)