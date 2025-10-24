import json

# 1. Load your responses from the JSON file
try:
    with open('my_responses.json', 'r', encoding='utf-8') as f:
        my_responses = json.load(f)
except FileNotFoundError:
    print("Error: 'my_responses.json' file not found.")
    print("Please create the file and add your triggers.")
    exit()
except json.JSONDecodeError:
    print("Error: 'my_responses.json' has a formatting error.")
    print("Please check for missing commas or quotes.")
    exit()

print("Your personal assistant is ready. (Type 'quit' to exit)")
print("--------------------------------------------------")

# 3. Start the chatbot loop
while True:
    # Get input from the user (and make it lowercase)
    user_input = input("You: ").lower()

    # Check if the user wants to quit
    if user_input == 'quit':
        print("Goodbye!")
        break

    response_found = False
    
    # --- MODIFICATION START ---
    # Loop through all the triggers you defined in your JSON
    for trigger in my_responses.keys():
        # Check if the trigger phrase is *inside* the user's input
        if trigger in user_input:
            print("\nAssistant says:")
            print(my_responses[trigger])
            print("--------------------------------------------------")
            response_found = True
            break  # Found a match, so stop looping
    # --- MODIFICATION END ---

    # If, after checking all keys, no response was found
    if not response_found:
        print("\nAssistant says:")
        print("I'm here to listen. (That trigger isn't defined.)")
        print("--------------------------------------------------")
