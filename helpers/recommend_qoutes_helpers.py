from bson import ObjectId
import ast
from datetime import datetime
import re

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


def getQuotationDetails(quotations, db):
    # Initialize a list to store the filtered details of quotations
    quotation_details = []

    for quote in quotations:
        # Ensure the key matches: change "Quote_Id" to "Quote_id"
        quote_id_raw = quote.get("Quote_id")  # Use correct key
        if not quote_id_raw:
            print(f"Missing Quote_id in quote: {quote}")
            continue

        # Ensure the Quote_id is a valid ObjectId
        try:
            quote_id = ObjectId(quote_id_raw) if isinstance(quote_id_raw, str) else quote_id_raw
        except Exception as e:
            print(f"Invalid ObjectId format: {quote_id_raw} - {e}")
            continue

        # Access the quotations collection
        quotation_collection = db.quotations

        # Query the collection to find the quotation by its ID
        result = quotation_collection.find_one(
            {"_id": quote_id},
            {"_id": 1, "item": 1, "quantity": 1, "price": 1}
        )

        if result:
            # Convert ObjectId to string for JSON compatibility
            result["id"] = str(result["_id"])  # Add stringified ID as "id"
            del result["_id"]  # Optionally remove the original "_id"

            # Add delivery_date and vendor from the input `quote`
            result["delivery_date"] = quote.get("delivery_date", "Not provided")
            result["vendor"] = quote.get("vendor", "Not provided")

            # Append the result to the quotation details
            quotation_details.append(result)
        else:
            print(f"No result found for Quote_id: {quote_id}")
    return quotation_details if quotation_details else None




def evaluate_quotes(items, criteria, weights=None):
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("The items list must contain at least one item.")

    # Convert delivery_date strings to datetime objects
    for item in items:
        if isinstance(item.get("delivery_date"), str):
            item["delivery_date"] = datetime.strptime(item["delivery_date"], "%Y-%m-%dT%H:%M:%S")

    # Handle the single-item case
    if len(items) == 1:
        single_item = items[0]
        return [{
            "quote": single_item,
            "criteria_matched": "Only item in the list",
        }]

    criteria = criteria.lower()
    if criteria not in ["price", "delivery", "balanced"]:
        raise ValueError('Criteria must be "price", "delivery", or "balanced".')

    if weights is None:
        weights = (0.5, 0.5)

    if not isinstance(weights, tuple) or len(weights) != 2:
        raise TypeError("Weights must be a tuple of two values (price_weight, date_weight).")

    price_weight, date_weight = weights
    if not (0 <= price_weight <= 1 and 0 <= date_weight <= 1 and price_weight + date_weight == 1):
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

    # Earliest Delivery Date (Handle ties)
    elif criteria == "delivery":
        earliest_date = min(item["delivery_date"] for item in items)
        best_date_items = [item for item in items if item["delivery_date"] == earliest_date]
        return [
            {
                "quote": item,
                "criteria_matched": "Earliest Delivery Date",
            }
            for item in best_date_items
        ]

    # Balanced Price and Delivery Date (Handle ties)
    elif criteria == "balanced":
        min_price = min(item["price"] for item in items)
        earliest_date = min(item["delivery_date"] for item in items)

        def weighted_score(item):
            normalized_price = (min_price / item["price"])  # Inverse normalization for price
            normalized_date = (earliest_date.timestamp() / item["delivery_date"].timestamp())  # Inverse normalization for date
            return (price_weight * normalized_price) + (date_weight * normalized_date)

        # Calculate scores for all items
        scores = [(item, weighted_score(item)) for item in items]
        max_score = max(score for _, score in scores)

        # Get all items with the maximum score
        best_balanced_items = [item for item, score in scores if score == max_score]
        return [
            {
                "quote": item,
                "criteria_matched": "Balanced Price and Delivery Date",
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
    # Initialize a list to store all quotations with details for all bids
    quotations_with_details_for_all_bids = []

    for bidId in bidIds:
        # Initialize a list to store quotations for the current bid
        quotations_with_details_one_bid = []

        # Ensure the bidId is a valid ObjectId (if it's a string, convert it)
        if isinstance(bidId, str):
            try:
                bidId = ObjectId(bidId)  # Convert to ObjectId from string
            except Exception as e:
                print(f"Invalid ObjectId format: {bidId} - {e}")
                continue

        # Access the bids collection
        bids_collection = db.bids

        # Query the bids collection to find the quotations, vendor, and delivery date for the given bidId
        result = bids_collection.find_one(
            {"_id": bidId},
            {"quotations": 1, "vendor": 1, "deliveryDate": 1, "_id": 0}
        )

        if result:
            # Extract the values
            delivery_date = result.get("deliveryDate", None)
            vendor = result.get("vendor", None)
            quotations = result.get("quotations", [])

            if quotations:
                # Create a dictionary for each quotation ID with delivery date and vendor
                for q in quotations:
                    quotation_detail = {
                        "Quote_id": str(q),  # Convert ObjectId to string
                        "delivery_date": delivery_date.isoformat() if delivery_date else None,  # Convert to ISO 8601 format
                        "vendor": get_user_name(vendor, db)  # Replace vendor ID with name
                    }
                    quotations_with_details_one_bid.append(quotation_detail)
            else:
                # Handle the case where no quotations exist
                quotations_with_details_one_bid.append({
                    "Quote_id": [],
                    "delivery_date": delivery_date.isoformat() if delivery_date else None,
                    "vendor": vendor
                })
        else:
            # If no result is found for the bidId, append a placeholder dictionary
            quotations_with_details_one_bid.append({
                "Quote_id": [],
                "delivery_date": None,
                "vendor": None
            })

        # Append the details for this bid to the overall list
        quotations_with_details_for_all_bids.extend(quotations_with_details_one_bid)

    # Return the list of dictionaries
    return quotations_with_details_for_all_bids

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
        print(f"Here it is:{type(purchase_request_id)}:{purchase_request_id} :  {type(request_id)}:{request_id}")


        # Get the title for the purchase request
        title = getPurchaseRequestTitle(purchase_request_id, db)
        print(title)


        # Append to final list
        final.append({"ID": request_id, "Title": title})


    return final

def get_user_name(user_id, db):
    try:
        # Find the user in the collection
        user = db.users.find_one({"_id": user_id}, {"firstName": 1, "lastName": 1, "_id": 0})
        if user:
            return user
        else:
            return {"error": "User not found"}
    except Exception as e:
        return {"error": str(e)}

def get_best_quotes(quotations):
    # Ensure input is not empty
    if not quotations:
        return {"lowest_price": [], "earliest_delivery_date": []}

    # Initialize variables
    lowest_price = float('inf')
    earliest_delivery_date = datetime.max
    quotes_with_lowest_price = []
    quotes_with_earliest_delivery = []

    # Process quotations
    for quote in quotations:
        price = quote.get('price', float('inf'))
        delivery_date_str = quote.get('delivery_date', None)
        delivery_date = datetime.fromisoformat(delivery_date_str) if delivery_date_str else datetime.max

        # Check for lowest price
        if price < lowest_price:
            lowest_price = price
            quotes_with_lowest_price = [quote]
        elif price == lowest_price:
            quotes_with_lowest_price.append(quote)

        # Check for earliest delivery date
        if delivery_date < earliest_delivery_date:
            earliest_delivery_date = delivery_date
            quotes_with_earliest_delivery = [quote]
        elif delivery_date == earliest_delivery_date:
            quotes_with_earliest_delivery.append(quote)

    # Return results
    return {
        "lowest_price": quotes_with_lowest_price,
        "earliest_delivery_date": quotes_with_earliest_delivery,
    }

def extract_number(user_input):
    # Use a regular expression to find the first number in the input string
    match = re.search(r'\d+', user_input)
    if match:
        return int(match.group())
    return None

def extract_keywords(user_input):
    # Define the keywords to match
    keywords = ["Price", "Delivery", "Balanced"]

    # Create a regular expression pattern to match any of the keywords
    pattern = r'\b(?:' + '|'.join(keywords) + r')\b'

    # Find all matches (case-insensitive)
    matches = re.findall(pattern, user_input, re.IGNORECASE)

    # Process matches
    if not matches:
        return None
    elif len(matches) == 1:
        return matches[0].capitalize()
    else:
        return [match.capitalize() for match in matches]


def check_status(input_text):
    # Define rejection-related and acceptance-related keywords
    rejection_keywords = ["cancel", "rejected", "reject", "decline", "disapprove", "deny"]
    acceptance_keywords = ["accept", "accepted", "approve", "approved", "confirm", "agree"]

    # Compile regex patterns for both groups (case-insensitive)
    rejection_pattern = r'\b(?:' + '|'.join(rejection_keywords) + r')\b'
    acceptance_pattern = r'\b(?:' + '|'.join(acceptance_keywords) + r')\b'

    # Check for matches in the input text
    if re.search(rejection_pattern, input_text, re.IGNORECASE):
        return "rejected"
    elif re.search(acceptance_pattern, input_text, re.IGNORECASE):
        return "accepted"
    else:
        return None


def check_review_response(user_input):

    # Define yes-like and no-like keywords (case-insensitive)
    yes_keywords = ["yes", "another", "sure", "okay", "yep", "yup", "affirmative", "yeah", "alright", "submit"]
    no_keywords = ["no", "not", "nope", "never", "nah", "negative", "none", "cancel"]

    # Create regex patterns for keyword matching
    yes_pattern = r'\b(?:' + '|'.join(yes_keywords) + r')\b'
    no_pattern = r'\b(?:' + '|'.join(no_keywords) + r')\b'

    # Check for yes-like keywords in the input
    if re.search(yes_pattern, user_input, re.IGNORECASE):
        return "yes"
    # Check for no-like keywords in the input
    elif re.search(no_pattern, user_input, re.IGNORECASE):
        return "no"
    # Return None if no match is found
    return None
