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
/* Hide all progress bars */
.progress-bar, .progress {
    display: none !important;
}

/* Overall styling */
body, .gradio-container {
    background-color: #1A1A1A !important;
    color: white;
}

#chatbot {
    background-color: #1A1A1A !important;
    border: none !important;
    padding: 0 !important;
}

.gr-message-bubble {
    border-radius: 16px !important;
    padding: 12px 16px !important;
    line-height: 1.6;
}

.gr-message-user {
    background-color: #007BFF !important;
    color: white !important;
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
}

/* Input textbox */
#input_textbox {
    width: 100%;
    border: none !important;
    background: transparent !important;
    color: white !important;
    padding: 12px 50px 12px 15px !important;
}

/* Send button */
#send_button {
    position: absolute !important;
    right: 8px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    background: #4a90e2 !important;
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
}

/* Loading spinner */
#send_button.loading {
    background: transparent !important;
    border: 3px solid rgba(255,255,255,0.2) !important;
    border-top: 3px solid #4a90e2 !important;
    animation: spin 1s linear infinite !important;
}

@keyframes spin {
    0% { transform: translateY(-50%) rotate(0deg); }
    100% { transform: translateY(-50%) rotate(360deg); }
}
"""

# --- Gradio UI Block ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")
    
    chatbot = gr.Chatbot(height=400, elem_id="chatbot")
    loading_state = gr.State(False)

    with gr.Column(elem_id="input_container"):
        msg = gr.Textbox(
            placeholder="Ask me anything about Akshay's profile...",
            elem_id="input_textbox",
            container=False
        )
        submit = gr.Button("➤", elem_id="send_button")

    def respond(message, chat_history, loading_state):
        # Clear input immediately
        yield "", chat_history, True
        
        # Get response
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(
            client, "llama-3.3-70b-versatile", message, relevant_excerpts
        )
        
        # Update chat
        chat_history.append((message, bot_message))
        yield "", chat_history, False

    # Set up event handlers
    submit.click(
        respond,
        [msg, chatbot, loading_state],
        [msg, chatbot, loading_state],
        queue=True
    )
    
    msg.submit(
        respond,
        [msg, chatbot, loading_state],
        [msg, chatbot, loading_state],
        queue=True
    )
    
    # Toggle loading spinner
    loading_state.change(
        lambda x: gr.Button(visible=not x),
        loading_state,
        submit,
        js="""
        function(start) {
            const btn = document.getElementById('send_button');
            if (start) {
                btn.classList.add('loading');
                btn.innerHTML = '';
            } else {
                btn.classList.remove('loading');
                btn.innerHTML = '➤';
            }
            return start;
        }
        """
    )

if __name__ == "__main__":
    demo.launch()