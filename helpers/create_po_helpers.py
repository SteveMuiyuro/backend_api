def find_quote_by_id(quotes, user_input_id):
    for quote in quotes:
        if quote.get("id") == user_input_id:
            return quote
    return None
