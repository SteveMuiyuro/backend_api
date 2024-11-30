from langchain.prompts import PromptTemplate

assign_workflow_greeting_template = PromptTemplate(
    template=" Start by greeting  {user_name}! introduce yourself as Sarm the user's assistant in assigning purchase request to a workflow. Ask the user if they have an Purcharse Request ID, if they do,request them provide it. If not, quote that you can can display recent Purchase Requests that have not been assigned into a workflow for the user."
)

assign_list_of_recent_requests_template = PromptTemplate(
    template="Analyze the {input} and based on the content, you can quote that the can select from a list recent purchase requests awaiting assinment to a workflow. Don't display the list here. request the user to kindly select input the ID of the purchase request you want to assign."
)

assign_unable_to_fetch_requests_template = PromptTemplate(
    template=" Respond that you are unable to fetch purchase requests due to the error {error} "
)

assign_purchase_requests_selected_template = PromptTemplate(
    template=" Respond that the purchase request:{input} has been selected. Request the user to enter a workflow they would wish to assign the request to , If tehy don't have a workflow in mind note that you can list  the available workflows that the user can select from"
)

assign_purchase_requests_list_workflows_template = PromptTemplate(
    template="Analyze the {input} and based on the content, you can quote that the user should select a valid approval workflows from the list available below. Keep it atmost two sentences"
)
