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

# --- Load Models & Data ---
embedding_model = setup_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2")
my_resume = load_text_data("data/resume.txt")
chunks = [chunk.strip() for chunk in my_resume.split("---") if chunk.strip()]
db = Chroma.from_texts(chunks, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Setup LLM (Groq) ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Custom CSS ---
custom_css = """
body, .gradio-container {
    background-color: #1A1A1A !important;
    color: white;
}

/* Hide all loading indicators */
.progress-bar, .animate-spin, .processing-time, [data-testid="progress-bar"], .clear-button {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}

/* Chat container */
#chatbot {
    background-color: #1A1A1A !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Intro section */
#intro_container {
    text-align: center;
    margin-top: 20px;
    color: #ccc;
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}

#intro_image {
    width: 120px;
    height: auto;
    border-radius: 50%;
    margin-bottom: 12px;
    object-fit: contain !important;
    display: block;
    margin-left: auto;
    margin-right: auto;
}


/* Message bubbles */
.gr-message-bubble {
    border-radius: 16px !important;
    padding: 12px 16px !important;
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

/* Input container */
#input_container {
    position: relative;
    background-color: #2C2C2C;
    border: 1px solid #444;
    border-radius: 25px;
    margin-top: 10px;
    margin-bottom: 20px;
    transition: border 0.2s ease;
}

#input_container:focus-within {
    border-color: #4a90e2;
}

/* Input box */
#input_textbox {
    width: 100%;
    border: none !important;
    background-color: transparent !important;
    color: #fff !important;
    font-size: 15px;
    padding: 12px 50px 12px 15px !important;
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

#input_textbox textarea::placeholder {
    color: #aaa;
    font-style: italic;
}

/* Send button */
#send_button {
    position: absolute !important;
    right: 8px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    background-color: #4a90e2 !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease !important;
}

#send_button:hover {
    background-color: #357ABD !important;
}
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")

    # 🌟 Intro (shown first)
    with gr.Column(visible=True, elem_id="intro_container") as intro_section:
        gr.Image(
            value="data/avatar.png",
            elem_id="intro_image",
            show_label=False,
            show_download_button=False,
            show_fullscreen_button=False,
            show_share_button=False,
            interactive=False
        )
        gr.Markdown("""
        Welcome to the Resume Chatbot for **Akshay Abraham**!  
        Ask me anything about Akshay's career, skills, or experiences.  
        Just type below and hit send.
        """)

    # 🤖 Chatbot (initially hidden)
    chatbot = gr.Chatbot(visible=False, type="messages", height=400, elem_id="chatbot")

    # 💬 Input area (always visible)
    with gr.Column(elem_id="input_container"):
        msg = gr.Textbox(
            label="",
            placeholder="Ask me anything about Akshay's profile...",
            container=False,
            elem_id="input_textbox",
            lines=1,
            max_lines=5
        )
        submit = gr.Button("➤", elem_id="send_button")

    # 📤 Handle response
    def respond(message, chat_history):
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(client, "llama-3.3-70b-versatile", message, relevant_excerpts)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history, gr.update(visible=False), gr.update(visible=True)

    # 📩 Bind button & Enter key
    submit.click(respond, [msg, chatbot], [msg, chatbot, intro_section, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, intro_section, chatbot])

# 🚀 Launch
if __name__ == "__main__":
    demo.launch()
