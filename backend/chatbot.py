import json

# 1. Load your responses from the JSON file
with open('my_responses.json', 'r') as f:
    my_responses = json.load(f)

print("Your personal assistant is ready. (Type 'quit' to exit)")
print("--------------------------------------------------")

# 3. Start the chatbot loop
while True:
    # Get input from the user (and make it lowercase to match keys)
    user_input = input("How are you feeling? ").lower()

    # Check if the user wants to quit
    if user_input == 'quit':
        print("Goodbye!")
        break

    # Check if the user's input is one of your triggers
    if user_input in my_responses:
        # If it is, print the response you defined
        print("\nAssistant says:")
        print(my_responses[user_input])
        print("--------------------------------------------------")
    else:
        # A default reply if it's not a trigger you've set
        print("\nAssistant says:")
        print("I'm here to listen. (That trigger isn't defined yet.)")
        print("--------------------------------------------------")
