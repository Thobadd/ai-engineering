# Phase 3 Day 1 - Working with LLMs via Claude API
from anthropic import Anthropic

# Initialize the client
client = Anthropic()

# Create a simple chatbot with conversation history
conversation_history = []

def chat(user_message):
    """Send a message and get a response from Claude"""
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful AI assistant. Be concise and direct.",
        messages=conversation_history
    )
    
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

# Test the chatbot
print("=== Claude Chatbot ===\n")

# Test 1: Simple question
response1 = chat("What is machine learning in one sentence?")
print(f"Q: What is machine learning in one sentence?\nA: {response1}\n")

# Test 2: Follow-up (uses conversation history)
response2 = chat("Can you give me a real-world example?")
print(f"Q: Can you give me a real-world example?\nA: {response2}\n")

# Test 3: Another follow-up
response3 = chat("How is that different from deep learning?")
print(f"Q: How is that different from deep learning?\nA: {response3}\n")

# Show conversation history
print("=== Conversation History ===")
for msg in conversation_history:
    print(f"{msg['role'].upper()}: {msg['content'][:100]}...")