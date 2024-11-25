from bson import ObjectId
import ast

def updateBidStatus(quoteId, user_input, db):
    # Fetch the quotation by its ID
    quote = db.quotations.find_one({"_id": ObjectId(quoteId)})
    if not quote:
        raise ValueError(f"Quotation with ID {quoteId} not found.")

    # Access the bid ID
    bid_id = quote['bid']

    # Determine the status based on user input
    valid_statuses = ["Accepted", "Rejected"]
    status = user_input.capitalize()  # Ensure the input is in the correct case
    if status not in valid_statuses:
        return "Invalid",status

    # Update the status of the specific bid
    result = db.bids.update_one(
        {"_id": ObjectId(bid_id)},  # Filter for the specific bid
        {"$set": {"status": status}}  # Update the status field
    )

    if result.modified_count == 0:
        return "Failed", status

    return "Success",status


def getQuotationDetails(quotationIds, db):
    # Initialize a list to store the filtered details of quotations
    quotation_details = []

    for quotation_id in quotationIds:
        # Ensure the quotation_id is a valid ObjectId (if it's a string, convert it)
        if isinstance(quotation_id, str):
            try:
                quotation_id = ObjectId(quotation_id)  # Convert to ObjectId from string
            except Exception as e:
                print(f"Invalid ObjectId format: {quotation_id} - {e}")
                continue

        # Access the quotations collection
        quotation_collection = db.quotations

        # Query the collection to find the quotation by its ID
        result = quotation_collection.find_one(
            {"_id": quotation_id},
            {"_id": 1, "item": 1, "quantity": 1, "price": 1}
        )

        if result:
            # Convert ObjectId to string for JSON compatibility
            result["id"] = str(result["_id"])  # Add stringified ID as "id"
            del result["_id"]  # Optionally remove the original "_id"
            quotation_details.append(result)
            return quotation_details
        else:
            return None


def evaluate_quotes(items, criteria, weights=None):
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("The items list must contain at least one item.")

    # Handle the single-item case
    if len(items) == 1:
        single_item = items[0]
        return [{
            "quote":single_item,
            "criteria_matched": "Only item in the list",
        }]

    criteria = criteria.lower()
    if criteria not in ["price", "quantity", "balanced"]:
        raise ValueError('Criteria must be "price", "quantity", or "balanced".')

    if weights is None:
        weights = (0.5, 0.5)

    if not isinstance(weights, tuple) or len(weights) != 2:
        raise TypeError("Weights must be a tuple of two values (price_weight, quantity_weight).")

    price_weight, quantity_weight = weights
    if not (0 <= price_weight <= 1 and 0 <= quantity_weight <= 1 and price_weight + quantity_weight == 1):
        raise ValueError("Weights must be between 0 and 1 and sum to 1.")

    # Lowest Price (Handle ties)
    if criteria == "price":
        min_price = min(item["price"] for item in items)
        best_price_items = [item for item in items if item["price"] == min_price]
        return [
            {
                "quote": item,
                "criteria_matched": "Lowest Price",
            }
            for item in best_price_items
        ]

    # Highest Quantity (Handle ties)
    elif criteria == "quantity":
        max_quantity = max(item["quantity"] for item in items)
        best_quantity_items = [item for item in items if item["quantity"] == max_quantity]
        return [
            {
                "quote": item,
                "criteria_matched": "Highest Quantity",
            }
            for item in best_quantity_items
        ]

    # Balanced Price and Quantity (Handle ties)
    elif criteria == "balanced":
        max_quantity = max(item["quantity"] for item in items)
        min_price = min(item["price"] for item in items)

        def weighted_score(item):
            normalized_price = (min_price / item["price"])  # Inverse normalization for price
            normalized_quantity = (item["quantity"] / max_quantity)
            return (price_weight * normalized_price) + (quantity_weight * normalized_quantity)

        # Calculate scores for all items
        scores = [(item, weighted_score(item)) for item in items]
        max_score = max(score for _, score in scores)

        # Get all items with the maximum score
        best_balanced_items = [item for item, score in scores if score == max_score]
        return [
            {
                "quote": item,
                "criteria_matched": "Balanced Price and Quantity",
            }
            for item in best_balanced_items
        ]


def find_request_id(input_value, db):
    try:
        rfq_input = int(input_value.strip())  # Convert to string for flexibility
        rfq = db.requestfors.find_one({"requestId": rfq_input}, {"requestId": 1, "_id": 0})

        if rfq:
            print(f"RFQ found: {rfq}")
            return rfq["requestId"]  # Return the actual requestId
        else:
            print(f"No RFQ found for requestId: {rfq_input}")
            return None

    except Exception as e:
        print(f"Error while querying RFQs: {str(e)}")
        return None



def getBids(requestForID, db):
    if isinstance(requestForID, str):
        try:
            requestForID = ast.literal_eval(requestForID)
        except (ValueError, SyntaxError):
            print(f"Invalid format for requestForID: {requestForID}")
            return []

    if isinstance(requestForID, dict) and 'requestId' in requestForID:
        requestForID = requestForID['requestId']

    try:
        newRequestId = int(requestForID)
    except ValueError:
        print(f"Invalid requestForID format: {requestForID}")
        return []

    request_document = db.requestfors
    results = request_document.find({"requestId": newRequestId}, {"bids": 1, "_id": 0})

    # Flatten the bids list and convert ObjectId to string
    bids = []
    for result in results:
        if "bids" in result:
            # Flatten nested arrays and convert ObjectId to string
            bid = result["bids"]
            if isinstance(bid, list):
                for bid_item in bid:
                    # Convert ObjectId to string if it's present
                    if isinstance(bid_item, ObjectId):
                        bids.append(str(bid_item))
                    else:
                        bids.append(bid_item)
    return bids

def getQuotationsForBids(bidIds, db):
    # Initialize an empty list to store the quotations for each bid
    all_quotations = []

    for bidId in bidIds:
        # Ensure the bidId is a valid ObjectId (if it's a string, convert it)
        if isinstance(bidId, str):
            try:
                bidId = ObjectId(bidId)  # Convert to ObjectId from string
            except Exception as e:
                print(f"Invalid ObjectId format: {bidId} - {e}")
                all_quotations.append("Invalid ObjectId format")
                continue

        bids_collection = db.bids

        # Query the bids collection to find the quotations for the given bidId
        result = bids_collection.find_one({"_id": bidId}, {"quotations": 1, "_id": 0})

        if result and 'quotations' in result:
            # If quotations exist for the bid, add them as a list of strings
            quotation_ids = [str(quotation_id) for quotation_id in result['quotations']]
            all_quotations.extend(quotation_ids)  # Add the quotations to the result list
        else:
            # If no quotations are found, add an empty list
            all_quotations.extend([])

    return all_quotations

def getPurchaseID(requestForID, db):
    request_document = db.requestfors
    result = request_document.find_one(
        {"_id": ObjectId(requestForID)},
        {"purchaseRequest": 1, "requestId": 1, "_id": 0}
    )
    if result:
        purchase_request = result.get("purchaseRequest")
        request_id = result.get("requestId")

        return {"purchaseRequest": purchase_request, "requestId": request_id}
    else:
        return None

def getPurchaseRequestTitle(purcharseRequestID, db):
    request_collection = db.requests
    result = request_collection.find_one({"_id":ObjectId(purcharseRequestID)}, {"title":1, "_id":0})
    if result:
        title = str(result["title"])
        return title
    else:
        return None

def listAllRFQs(db):
    bids_collection = db.bids
    bid_lists =  bids_collection.find({}, {'_id': 0, 'request': 1})

    # Build the final list with requestId and title
    final = []
    for doc in bid_lists:
        purchase_request = doc["request"]


        # Get the purchase request ID and requestId
        purchase_data = getPurchaseID(purchase_request, db)
        request_id = purchase_data["requestId"]
        purchase_request_id = purchase_data["purchaseRequest"]

        # Get the title for the purchase request
        title = getPurchaseRequestTitle(purchase_request_id, db)

        # Append to final list
        final.append({"ID": request_id, "Title": title})
    print(final)


    return final
