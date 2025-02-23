import gradio as gr
from openai_chat import chat_fn

def respond(message, history, model):
    """Respond to a new user message using the selected OpenAI model.
    
    Args:
        message (str): The current user message.
        history (list): The current conversation history (list of dicts).
        model (str): The model selected from the dropdown.
        
    Returns:
        A tuple containing:
          - An empty string (to clear the textbox),
          - The updated conversation state,
          - The updated chat display.
    """
    reply = chat_fn(message, history, model)
    
    if history is None:
        history = []
    
    # Append the new exchange as a dictionary for user and assistant respectively.
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ]
    
    return "", history, history

def clear_chat(model):
    """
    Automatically clears the chat history when the model is changed.
    
    Args:
        model (str): The new model (unused for clearing).
    
    Returns:
        A tuple resetting the textbox, conversation state, and chat display.
    """
    return "", [], []

with gr.Blocks() as demo:
    gr.Markdown("# OpenAI Chatbot with Model Selection")
    
    # Model selection dropdown.
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["gpt-4o", "o3-mini"],
            value="gpt-4o",
            label="Select Model"
        )
    
    # Chat components: using type="messages" to follow the openai-style dictionaries.
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    state = gr.State([])
    
    # Submit user message -> call respond().
    msg.submit(respond, inputs=[msg, state, model_dropdown], outputs=[msg, state, chatbot])
    
    # Change model -> clear chat.
    model_dropdown.change(clear_chat, inputs=model_dropdown, outputs=[msg, state, chatbot])
    
if __name__ == "__main__":
    demo.launch()

