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
/* MAIN BACKGROUND */
body, .gradio-container, .dark, #chatbot {
    background-color: #1A1A1A !important;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Hide scrollbars */
#chatbot::-webkit-scrollbar {
    display: none !important;
}
#chatbot {
    -ms-overflow-style: none !important;
    scrollbar-width: none !important;
}

/* Lottie container */
#lottie_container {
    background: transparent !important;
    border: none !important;
    margin: 0 auto;
    width: 250px !important;
    height: 250px !important;
}

/* Chat interface */
#chatbot {
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Message bubbles */
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
    margin: 20px auto;
    max-width: 600px;
}

/* Input box */
#input_textbox {
    width: 100%;
    border: none !important;
    background: transparent !important;
    color: #fff !important;
    padding: 12px 50px 12px 15px !important;
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
}
"""

# --- JavaScript for auto-scrolling ---
js = """
function scrollToBottom() {
    const chatbot = document.querySelector('#chatbot');
    chatbot.scrollTop = chatbot.scrollHeight;
    return [];
}
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css, js=js) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")

    # Lottie Animation Intro
    with gr.Column(visible=True, elem_id="intro_container") as intro_section:
        gr.HTML("""
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <div id="lottie_container">
            <dotlottie-player src="https://lottie.host/3a69db62-ac6b-419d-8949-79fe213690c8/QJbL66mr48.lottie" 
            background="transparent" speed="1" style="width:100%;height:100%" loop autoplay></dotlottie-player>
        </div>
        """)
        gr.Markdown("Hello! I'm your AI assistant 🤖<br>Ask me anything about Akshay's professional background!")

    # Chat Interface
    chatbot = gr.Chatbot(visible=False, elem_id="chatbot", height=500)
    
    with gr.Row(elem_id="input_container"):
        msg = gr.Textbox(
            placeholder="Type your question here...", 
            show_label=False, 
            container=False,
            elem_id="input_textbox",
            autofocus=True
        )
        submit = gr.Button("➤", elem_id="send_button")

    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        
        # Process message
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(client, "llama3-70b", message, relevant_excerpts)
        
        # Update chat history
        chat_history.append((message, bot_message))
        
        return "", chat_history, gr.update(visible=False), gr.update(visible=True)

    # Bind interactions
    submit.click(
        respond,
        [msg, chatbot],
        [msg, chatbot, intro_section, chatbot],
        _js=js,
        api_name="send_message"
    )
    
    msg.submit(
        respond,
        [msg, chatbot],
        [msg, chatbot, intro_section, chatbot],
        _js=js,
        api_name="submit_message"
    )

if __name__ == "__main__":
    demo.launch()