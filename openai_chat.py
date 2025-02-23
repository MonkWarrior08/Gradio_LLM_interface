from openai import OpenAI
from dotenv import load_dotenv
import os


# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI client using the provided API key.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_fn(message, history, model="gpt-4o", prompt="You are a helpful assistant."):
    """
    Processes a chat message using the OpenAI API with streaming enabled. 
    Yields the assistant's reply token-by-token.

    Args:
        message (str): The current user message.
        history (list): The conversation history as a list of dictionaries.
        model (str): The OpenAI model to use.
        prompt (str): The system prompt to use.

    Yields:
        str: The assistant's reply accumulated so far.
    """
    if history is None:
        history = []
    
    # Build the conversation starting with the selected system prompt.
    conversation = [{"role": "system", "content": prompt}]
    conversation.extend(history)
    conversation.append({"role": "user", "content": message})
    
    # Call the OpenAI API with streaming enabled.
    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        stream=True
    )
    
    partial_reply = ""
    for chunk in response:
        # Directly access the 'content' attribute of the delta object.
        content = chunk.choices[0].delta.content
        if content:
            partial_reply += content
        yield partial_reply
