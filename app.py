import gradio as gr
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
/* Main Background */
body, .gradio-container, .dark {
    background-color: #1A1A1A !important;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Hide Scrollbars */
#chatbot::-webkit-scrollbar {
    width: 0 !important;
    background: transparent !important;
}

/* Loading Animation */
@keyframes spin {
    to { transform: rotate(360deg); }
}

#loading_icon {
    width: 36px;
    height: 36px;
    border: 3px solid rgba(74, 144, 226, 0.3);
    border-radius: 50%;
    border-top-color: #4a90e2;
    animation: spin 1s linear infinite;
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    display: none;
}

/* Lottie container */
#lottie_container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin: 0 auto;
    width: 250px !important;
    height: 250px !important;
}

/* Chat Interface */
#chatbot {
    background-color: #1A1A1A !important;
    border: none !important;
    padding: 0 !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

/* Message Bubbles */
.gr-message-bubble {
    border-radius: 16px !important;
    padding: 12px 16px !important;
    font-size: 15px;
    margin: 8px 0;
    opacity: 0;
    animation: messageAppear 0.4s forwards;
    max-width: 80%;
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

/* Input Container */
#input_container {
    position: relative;
    background-color: #2C2C2C;
    border: 1px solid #444;
    border-radius: 25px;
    margin: 20px auto;
    max-width: 600px;
}

/* Send Button */
#send_button {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background-color: #4a90e2;
    color: white;
    border: none;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
}

#send_button:hover {
    background-color: #357ABD;
    transform: translateY(-50%) scale(1.1);
}
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")

    # Lottie Animation
    with gr.Column(visible=True, elem_id="intro_container") as intro_section:
        gr.HTML("""
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <div id="lottie_container">
            <dotlottie-player
                src="https://lottie.host/3a69db62-ac6b-419d-8949-79fe213690c8/QJbL66mr48.lottie"
                background="transparent"
                speed="1"
                style="width: 100%; height: 100%"
                loop
                autoplay>
            </dotlottie-player>
        </div>
        """)
        gr.Markdown("""
        <div style='animation: fadeIn 0.8s ease-out;'>
        Hello! I'm your AI assistant 🤖<br>
        Ready to explore Akshay's professional background!
        </div>
        """)

    chatbot = gr.Chatbot(visible=False, type="messages", height=400, elem_id="chatbot")

    with gr.Column(elem_id="input_container"):
        msg = gr.Textbox(
            label="",
            placeholder="Ask me anything about Akshay's profile...",
            container=False,
            elem_id="input_textbox",
            lines=1,
            max_lines=5
        )
        with gr.Row():
            submit = gr.Button("➤", elem_id="send_button", visible=True)
            loading_icon = gr.HTML("<div id='loading_icon'></div>", visible=False)

    def respond(message, chat_history):
        # Clear input and show loading immediately
        yield "", chat_history, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
        
        # Process message
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(
            client, 
            "llama-3.3-70b-versatile", 
            message, 
            relevant_excerpts
        )
        
        # Update chat and restore UI
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        yield "", chat_history, gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

    # Bind interactions
    submit.click(
        respond,
        [msg, chatbot],
        [msg, chatbot, intro_section, chatbot, submit, loading_icon],
        queue=False
    )
    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot, intro_section, chatbot, submit, loading_icon],
        queue=False
    )

if __name__ == "__main__":
    demo.launch()