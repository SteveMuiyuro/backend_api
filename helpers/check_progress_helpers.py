from bson.objectid import ObjectId

def get_request_details(request_id, db):
    # Fetch the main request dictionary
    request = db.requests.find_one({"_id": ObjectId(request_id)})
    if not request:
        return {"error": "Request not found"}

    # Helper function to fetch detailed action information
    def fetch_action_details(action_id):
        action = db.actions.find_one({"_id": ObjectId(action_id)})
        if action:
            return {
                "action": action.get("action"),
                "status": action.get("status"),
                "created_at": action.get("createdAt"),
                "scheduled": action.get("scheduled", False),  # Default False if not present
            }
        return {"error": f"Action with ID {str(action_id)} not found"}

    # Process actions: Replace IDs with detailed dictionaries
    actions_details = [fetch_action_details(action_id) for action_id in request.get("actions", [])]

    # Build the final outer dictionary
    result = {
        "status": request.get("status"),
        "workflow": str(request.get("workflow")) if request.get("workflow") else None,
        "approving_department": str(request.get("approvingDepartment")) if request.get("approvingDepartment") else None,
        "actions": actions_details,  # List of detailed dictionaries for each action
        "created_at": request.get("createdAt"),
        "department": str(request.get("department")) if request.get("department") else None,
    }

    return result



# def get_action_details(action_id, db):
#     try:
#         # Query the database for the given action ID
#         action = db.actions.find_one({"_id": ObjectId(action_id)}, {
#             "action": 1,
#             "status": 1,
#             "createdAt": 1,
#             "scheduled": 1
#         })

#         if not action:
#             return {"error": f"Action ID {action_id} not found"}

#         # Format the result
#         result = {
#             "action": action.get("action"),
#             "status": action.get("status"),
#             "created_at": action.get("createdAt"),
#             "scheduled": action.get("scheduled", False)  # Default to False if not present
#         }
#         return result

#     except Exception as e:
#         return {"error": str(e)}


# def get_actions_details(action_ids, db):
#     results = []
#     for action_id in action_ids:
#         details = get_action_details(action_id, db)
#         results.append(details)
#     return results
