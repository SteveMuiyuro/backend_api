import random
import string
from datetime import datetime

def generate_random_string(length=5):
    # Define the characters that can appear in the string: uppercase letters and digits
    characters = string.ascii_uppercase + string.digits

    # Use random.choices() to randomly choose `length` characters from the allowed characters
    random_string = ''.join(random.choices(characters, k=length))
    return random_string


def detect_priority(user_input):
    # Define priority levels and their associated synonyms
    priorities = {
        "urgent": ["urgent", "immediate", "asap", "critical", "emergency"],
        "high": ["high", "important", "pressing", "essential"],
        "medium": ["medium", "moderate", "average"],
        "low": ["low", "minimal", "less important", "non-urgent"]
    }

    # Normalize the input to lowercase for case-insensitive comparison
    normalized_input = user_input.lower()

    # Check the input for each priority level's synonyms
    for priority, synonyms in priorities.items():
        if any(word in normalized_input for word in synonyms):
            return priority

    # Default return value if no priority words are found
    return None

# Function to validate input
def validate_input(current_step, user_input, invalid_date_template, invalid_format_template, invalid_priority_template, conversation_chain):
    if current_step == "due_date":
        try:
            due_date = datetime.strptime(user_input, '%Y-%m-%d')
            if due_date.date() < datetime.now().date():
                return False, conversation_chain.predict(
            input=invalid_date_template.format())
        except ValueError:
            return False, conversation_chain.predict(
            input=invalid_format_template.format())
    elif current_step == "priority":
        user_input = detect_priority(user_input)
        if not user_input:
            return False, conversation_chain.predict(
            input=invalid_priority_template.format())
    return True, None
