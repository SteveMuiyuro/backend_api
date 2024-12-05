from bson.objectid import ObjectId
from datetime import datetime


def get_request_details(request_id, db):
    # Fetch the main request dictionary
    request = db.requests.find_one({"requestId": request_id})
    if not request:
        return {"error": "Request not found"}

    # Helper function to fetch detailed action information
    def fetch_action_details(action_id):
        action = db.actions.find_one({"_id": ObjectId(action_id)})
        if action:
            return {
                "action": action.get("action"),
                "status": action.get("status"),
                "created_at": action.get("createdAt").isoformat() if action.get("createdAt") else None,
                "scheduled": action.get("scheduled", False),  # Default False if not present
            }
        return {"error": f"Action with ID {str(action_id)} not found"}

    # Process actions: Replace IDs with detailed dictionaries
    actions_details = [fetch_action_details(action_id) for action_id in request.get("actions", [])]

    # Convert created_at to a string format
    created_at = request.get("createdAt")
    created_at_str = created_at.isoformat() if isinstance(created_at, datetime) else None
    workflow_id =  str(request.get("workflow")) if request.get("workflow") else None
    approving_department_id =  str(request.get("approvingDepartment")) if request.get("approvingDepartment") else None
    creating_department_id = str(request.get("department")) if request.get("department") else None
    workflow_name = get_workflow_name_by_id(workflow_id, db) if workflow_id else None
    creating_department_name = get_department_name_by_id(creating_department_id, db) if creating_department_id else None
    approving_department_name = get_department_name_by_id(approving_department_id, db) if approving_department_id else None
    if isinstance(workflow_id, tuple):
        workflow_id = workflow_id[0]


    # Build the final outer dictionary
    result = {
        "status": request.get("status"),
        "title": request.get("title"),
        "workflow": workflow_name,
        "approving_department": approving_department_name ,
        "actions": actions_details,  # List of detailed dictionaries for each action
        "created_at": created_at_str,
        "department": creating_department_name,
    }

    return result


def get_workflow_name_by_id(workflow_id, db):
    try:
        workflow = db.workflows.find_one({"_id": ObjectId(workflow_id)}, {"name": 1, "_id": 0})
        if workflow:
            return workflow.get("name", "Name not available")
        else:
            return "Workflow not found"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_department_name_by_id(department_id, db):
    try:
        department = db.departments.find_one({"_id": ObjectId(department_id)}, {"name": 1, "_id": 0})

        # Check if a department was found and return the name
        if department:
            return department.get("name", "Name not available")
        else:
            return "Department not found"
    except Exception as e:
        return f"An error occurred: {str(e)}"
