
from pymongo import DESCENDING

def find_Purchase_request_id(input_value, db):
    try:
        # Ensure input_value is stripped and treated as a string
        request_input = input_value.strip()
        request_id = db.requests.find_one({"requestId": request_input}, {"requestId": 1, "_id": 0})

        if request_id:
            print(f"Purchase request found: {request_id}")
            return request_id["requestId"]  # Return the actual requestId
        else:
            print(f"No purchase request found for requestId: {request_input}")
            return None

    except Exception as e:
        print(f"Error while querying purchase requests: {str(e)}")
        return None



def get_latest_pending_requests(db):
    try:
        # Query the collection to find pending requests, project specific fields, and sort by createdAt
        requests = (
            db.requests.find(
                {"status": "submitted"},  # Filter by pending status
                {"requestId": 1, "title": 1, "_id": 0}  # Project requestId and title
            )
            .sort("createdAt", DESCENDING)  # Sort by createdAt in descending order
            .limit(5)  # Limit to the latest 5 results
        )
        return list(requests)
    except Exception as e:
        print(f"An error occurred while fetching requests: {e}")
        return []

def get_distinct_workflows(db):
    try:
        # Query the workflows collection for distinct names
        distinct_workflows = db.workflows.distinct("name")
        return distinct_workflows
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
