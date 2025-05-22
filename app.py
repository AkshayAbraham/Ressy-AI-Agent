import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.vectorstores import Chroma
from utils import (
    load_text_data,
    resume_chat_completion,
    semantic_search,
    setup_embedding_model,
)
import os
from groq import Groq
from dotenv import load_dotenv

# Create cache directory if it doesn't exist
os.makedirs('.gradio/cached_examples', exist_ok=True)

# --- Setup Embedding Model ---
embedding_model = setup_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2")

# --- Load Text Data and Chunking ---
my_resume = load_text_data("data/resume.txt")
chunks = [chunk.strip() for chunk in my_resume.split("---") if chunk.strip()]

# --- Create a Chroma database ---
db = Chroma.from_texts(chunks, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Setting up the LLM (Groq API) ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Custom CSS for Styling ---
custom_css = """
/* Overall background color for the body and the main Gradio container */
body, .gradio-container {
    background-color: #1A1A1A !important;
    color: white;
}

/* Chatbot message area - blend with background */
#chatbot {
    background-color: #1A1A1A !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Message bubbles */
.gr-message-bubble {
    border-radius: 16px !important;
    padding: 12px 16px !important;
    line-height: 1.6;
    font-size: 15px;
    font-family: 'Segoe UI', sans-serif;
}

.gr-message-user {
    background-color: #007BFF !important;
    color: white !important;
    align-self: flex-end;
}

.gr-message-bot {
    background-color: #2C2C2C !important;
    color: white !important;
}

/* Input row container */
#input_row {
    position: relative;
    display: flex;
    align-items: center;
    background-color: #2C2C2C;
    border: 1px solid #444;
    border-radius: 25px;
    padding: 8px 15px;
    margin-top: 10px;
    margin-bottom: 20px;
    transition: border 0.2s ease;
}

#input_row:focus-within {
    border-color: #4a90e2;
}

/* Input textbox */
#input_textbox {
    flex-grow: 1;
    border: none !important;
    background-color: transparent !important;
    color: #fff !important;
    font-size: 15px;
    padding: 8px 45px 8px 15px !important;
    margin: 0 !important;
    min-height: 20px !important;
    box-shadow: none !important;
}

#input_textbox textarea {
    background-color: transparent !important;
    color: white !important;
    resize: none !important;
    border: none !important;
    outline: none !important;
    padding: 0 !important;
    font-family: 'Segoe UI', sans-serif;
    margin: 0 !important;
    min-height: 20px !important;
    max-height: 120px !important;
}

/* Placeholder */
#input_textbox textarea::placeholder {
    color: #aaa;
    font-style: italic;
}

/* Send button - perfectly circular */
#send_button {
    position: absolute !important;
    right: 10px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    background-color: #4a90e2 !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease !important;
}

#send_button:hover {
    background-color: #357ABD !important;
}

/* Button icon */
#send_button svg {
    width: 20px !important;
    height: 20px !important;
}

/* Hide default clear */
.clear-button {
    display: none !important;
}
"""

# --- Gradio UI Block ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")
    gr.Markdown("""
    ## About this Chatbot
    This is a Retrieval-Augmented Generation (RAG) chatbot powered by AI that allows you to interactively explore Akshay Abraham's professional profile.
    - **Technology**: Utilizes advanced semantic search and a powerful language model (via Groq API).
    - **Purpose**: Provide detailed, context-aware answers about Akshay's professional background, skills, and achievements.
    - **How it works**:
        1. Your question is semantically searched against resume chunks.
        2. Relevant excerpts are retrieved from Akshay's profile.
        3. A language model (Llama 3 70B hosted on Groq) generates a precise, contextual response.
    """)

    chatbot = gr.Chatbot(type="messages", height=400, elem_id="chatbot")

    with gr.Row(elem_id="input_row"):
        msg = gr.Textbox(
            label="",
            placeholder="Ask me anything about Akshay's profile...",
            container=False,
            elem_id="input_textbox",
            lines=1,
            max_lines=5
        )
        submit = gr.Button("➤", elem_id="send_button")

    def respond(message, chat_history):
        """
        Gradio function for chatbot interaction.
        Args:
            message (str): The user's question.
            chat_history (list): The chat history.
        Returns:
            tuple: Updated chat history and cleared textbox
        """
        # Perform semantic search to get relevant context from resume
        relevant_excerpts = semantic_search(message, retriever)

        # Get the LLM response using Groq API
        bot_message = resume_chat_completion(
            client, "llama-3.3-70b-versatile", message, relevant_excerpts
        )

        # Append to history and return both history and empty string for textbox
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()