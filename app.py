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

# Create cache directory
os.makedirs('.gradio/cached_examples', exist_ok=True)

# Setup Embedding Model
embedding_model = setup_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2")

# Load and chunk resume text
my_resume = load_text_data("data/resume.txt")
chunks = [chunk.strip() for chunk in my_resume.split("---") if chunk.strip()]

# Create Chroma vector database
db = Chroma.from_texts(chunks, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setup Groq client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Custom Gemini-style CSS ---
custom_css = """
body, .gradio-container {
    background-color: #1A1A1A !important;
    color: white;
}

#chatbot {
    background-color: #1A1A1A !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

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

/* Gemini-style input field with embedded send button */
#input_container {
    position: relative;
    width: 100%;
    margin-top: 12px;
}

#input_textbox {
    width: 100%;
    padding-right: 45px !important;
    border-radius: 25px;
    background-color: #2c2c2c !important;
    color: white !important;
    border: 1px solid #555;
    font-size: 15px;
    resize: none;
}

#input_textbox textarea {
    background: transparent !important;
    color: white !important;
    font-size: 15px;
    padding: 12px 16px;
    border-radius: 25px;
    line-height: 1.5;
    font-family: 'Segoe UI', sans-serif;
    border: none;
    outline: none;
}

#input_textbox textarea::placeholder {
    color: #aaa;
    font-style: italic;
}

/* Send icon inside textbox */
#send_button {
    position: absolute;
    top: 50%;
    right: 12px;
    transform: translateY(-50%);
    background-color: #4a90e2 !important;
    color: white !important;
    border: none !important;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    cursor: pointer;
    box-shadow: 0 0 4px rgba(0, 0, 0, 0.3);
}

#send_button:hover {
    background-color: #357ABD !important;
}

.clear-button {
    display: none !important;
}
"""

# --- Gradio UI ---
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

    # Gemini-style input container
    with gr.Box(elem_id="input_container"):
        msg = gr.Textbox(
            label="",
            placeholder="Ask me anything about Akshay's profile...",
            container=False,
            elem_id="input_textbox"
        )
        submit = gr.Button(value="➤", elem_id="send_button")

    # Bot logic
    def respond(message, chat_history):
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(
            client, "llama-3.3-70b-versatile", message, relevant_excerpts
        )
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
