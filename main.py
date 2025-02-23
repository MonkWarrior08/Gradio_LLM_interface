import gradio as gr
from openai_chat import chat_fn

# Map models to their available system prompt options.
prompt_options = {
    "gpt-4o": [
        "You are a helpful assistant.",
        "You speak in french.",
        "you are a pirate."
    ],
    "o3-mini": [
        "You are a helpful assistant",
        "you speak in chinese"
    ]
}

def respond(message, history, model, prompt):
    """
    Responds to a new user message and streams the assistant's reply.

    Args:
        message (str): The current user message.
        history (list): The current conversation history.
        model (str): The selected model.
        prompt (str): The system prompt.

    Yields:
        A tuple of outputs: (textbox_value, updated_history, updated_chat_display).
    """
    if history is None:
        history = []
    
    # Create an updated history that shows the new user message.
    updated_history = history + [{"role": "user", "content": message}]
    # Append an empty assistant response so that we can update it incrementally.
    updated_history_with_assistant = updated_history + [{"role": "assistant", "content": ""}]
    
    # Send an initial update with the new messages.
    yield "", updated_history_with_assistant, updated_history_with_assistant
    
    # Stream the assistant's reply token-by-token.
    # Notice we pass the original history (without the appended user message)
    # because chat_fn itself appends the new user message.
    for partial_reply in chat_fn(message, history, model, prompt):
        updated_history_with_assistant[-1]["content"] = partial_reply
        yield "", updated_history_with_assistant, updated_history_with_assistant

def clear_chat_and_update_prompt(model):
    """
    Clears the chat history and updates the prompt dropdown when the model is changed.
    
    Args:
        model (str): The new model.
    
    Returns:
        A tuple resetting the textbox, conversation state, chat display,
        and updating the prompt dropdown using gr.update().
    """
    new_prompts = prompt_options.get(model, ["You are a helpful assistant."])
    # Use gr.update() to update the Dropdown component.
    return "", [], [], gr.update(choices=new_prompts, value=new_prompts[0])

with gr.Blocks() as demo:
    gr.Markdown("# LLM-interface")
    
    # Row for model and prompt selection.
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["gpt-4o", "o3-mini"],
            value="gpt-4o",
            label="Select Model"
        )
        prompt_dropdown = gr.Dropdown(
            choices=prompt_options["gpt-4o"],
            value=prompt_options["gpt-4o"][0],
            label="Select Prompt"
        )
    
    # Chat components: the chatbot displays a history of exchanges.
    chatbot = gr.Chatbot(type="messages", height=800)
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    state = gr.State([])
    
    # When the user presses enter in the textbox, call respond() with the message,
    # the current state, the selected model, and the selected prompt.
    msg.submit(respond, inputs=[msg, state, model_dropdown, prompt_dropdown],
               outputs=[msg, state, chatbot])
    
    # When the model selection changes, clear the chat and update the prompt options.
    model_dropdown.change(clear_chat_and_update_prompt, inputs=model_dropdown,
                          outputs=[msg, state, chatbot, prompt_dropdown])
    
if __name__ == "__main__":
    demo.launch()

