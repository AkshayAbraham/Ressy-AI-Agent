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
body, .gradio-container {
    background-color: #1A1A1A !important;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Hide all loading indicators */
.progress-bar, .animate-spin, .processing-time, 
[data-testid="progress-bar"], .progress, 
.spinner, .loading, .clear-button {
    display: none !important;
}

/* Hide Scrollbars */
#chatbot::-webkit-scrollbar {
    width: 0 !important;
    background: transparent !important;
}

/* Lottie container styling */
#lottie_container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin: 0 auto;
    width: 250px !important;
    height: 250px !important;
}

dotlottie-player {
    background-color: #1A1A1A !important;
}

/* Chatbot container */
#chatbot {
    background-color: #1A1A1A !important;
    border: none !important;
    padding: 0 !important;
    transition: opacity 0.3s ease;
    scrollbar-width: none !important; /* Firefox */
    -ms-overflow-style: none !important; /* IE/Edge */
    overflow-y: auto !important; /* Maintain scroll functionality */

}

/* Intro section */
#intro_container {
    text-align: center;
    margin: 20px auto;
    max-width: 500px;
    animation: fadeIn 0.8s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Message bubbles */
.gr-message-bubble {
    border-radius: 16px !important;
    padding: 12px 16px !important;
    font-size: 15px;
    margin: 8px 0;
    opacity: 0;
    animation: messageAppear 0.4s forwards;
    max-width: 80%;
}

@keyframes messageAppear {
    to { opacity: 1; }
}

.gr-message-user {
    background-color: #007BFF !important;
    color: white !important;
    align-self: flex-end;
    animation-delay: 0.1s;
}

.gr-message-bot {
    background-color: #2C2C2C !important;
    color: white !important;
    animation-delay: 0.2s;
}

/* Input container */
#input_container {
    position: relative;
    background-color: #2C2C2C;
    border: 1px solid #444;
    border-radius: 25px;
    margin: 20px auto;
    max-width: 600px;
    transition: all 0.3s ease;
}

#input_container:focus-within {
    border-color: #4a90e2;
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
}

/* Input box */
#input_textbox {
    width: 100%;
    border: none !important;
    background: transparent !important;
    color: #fff !important;
    padding: 12px 50px 12px 15px !important;
    min-height: 20px !important;
}

#input_textbox textarea {
    background: transparent !important;
    resize: none !important;
    border: none !important;
    outline: none !important;
    padding: 0 !important;
    margin: 0 !important;
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
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

#send_button:hover {
    background-color: #357ABD !important;
    transform: translateY(-50%) scale(1.1) !important;
}
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")

    # 🤖 Lottie Animation Intro
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

    # 💬 Chatbot
    chatbot = gr.Chatbot(visible=False, type="messages", height=400, elem_id="chatbot")

    # ⌨️ Input area
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

    # 📤 Response handler
    def respond(message, chat_history):
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(
            client, 
            "llama-3.3-70b-versatile", 
            message, 
            relevant_excerpts
        )
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history, gr.update(visible=False), gr.update(visible=True)

    # 📩 Bind interactions
    submit.click(
        respond, 
        [msg, chatbot], 
        [msg, chatbot, intro_section, chatbot],
        show_progress=False
    )
    msg.submit(
        respond, 
        [msg, chatbot], 
        [msg, chatbot, intro_section, chatbot],
        show_progress=False
    )

# 🚀 Launch
if __name__ == "__main__":
    demo.launch()