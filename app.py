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
    height: 100vh !important;
    overflow: hidden !important;
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
    width: 120px !important;
    height: 120px !important;
    margin-bottom: 10px !important;
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
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
    overflow-y: auto !important;
    height: calc(100vh - 400px) !important;
}

/* Compact Intro Section */
#intro_container {
    text-align: center;
    margin: 0 auto;
    padding: 20px 0;
    max-width: 500px;
    height: calc(100vh - 180px);
    display: flex !important;
    flex-direction: column;
    justify-content: center;
}

/* Prompt Containers Layout */
.prompt-row {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin: 10px 0;
}

.prompt-container {
    background-color: #282828;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    border: 1px solid #3a3a3a;
    text-align: center;
    min-width: 160px;
    max-width: 180px;
    transition: all 0.3s ease;
    cursor: pointer;
}

.prompt-container:hover {
    background-color: #383838;
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}

.prompt-container p {
    margin: 0;
    color: #ffffff;
    font-size: 0.9em;
    line-height: 1.4;
}

.single-prompt {
    margin: 0 auto;
}

/* Fixed Input Container */
#input_container {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 90%;
    max-width: 600px;
    background-color: #2C2C2C;
    border: 1px solid #444;
    border-radius: 25px;
    transition: all 0.3s ease;
    z-index: 1000;
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

@keyframes messageAppear {
    to { opacity: 1; }
}

/* Top Icons */
.top-icons {
    display: flex;
    justify-content: flex-end;
    gap: 30px;
    padding: 10px 20px;
    position: fixed;
    top: 0;
    right: 0;
    z-index: 1000;
}

.top-icons button,
.top-icons a {
    background: none;
    border: none;
    cursor: pointer;
    transition: transform 0.3s ease;
}

.top-icons button:hover svg,
.top-icons a:hover svg {
    transform: scale(1.3) rotate(5deg);
}

.top-icons svg {
    fill: #ffffff;
    width: 28px;
    height: 28px;
    transition: transform 0.3s ease;
}

/* Info Modal */
#info_modal {
    display: none;
    position: fixed;
    top: 20%;
    left: 50%;
    transform: translateX(-50%);
    background: #2C2C2C;
    color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    z-index: 9999;
    max-width: 400px;
}

#info_modal h3 {
    color: #61dafb;
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 1.5em;
}

#info_modal p {
    font-size: 1em;
    line-height: 1.6;
    margin-bottom: 20px;
    color: #d0d0d0;
}

#info_modal button#close_modal {
    margin-top: 12px;
    background-color: #4a90e2;
    color: white;
    border: none;
    padding: 10px 16px;
    border-radius: 20px;
    font-size: 14px;
    cursor: pointer;
    transition: background-color 0.2s ease, transform 0.2s ease;
}

#info_modal button#close_modal:hover {
    background-color: #357ABD;
    transform: scale(1.05);
}

/* Mobile Responsiveness */
@media (max-height: 700px) {
    #lottie_container {
        height: 80px !important;
        width: 80px !important;
    }
    .prompt-container {
        padding: 10px;
        min-width: 140px;
    }
    .prompt-container p {
        font-size: 0.8em;
    }
}
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
<!-- Top Right Icons -->
<div class="top-icons">
    <button id="info_icon" title="About Agent">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M11 9h2V7h-2v2zm0 8h2v-6h-2v6zm1-16C5.48 1 1 5.48 1 11s4.48 10 10 10 10-4.48 10-10S16.52 1 12 1zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
        </svg>
    </button>
    <a id="download_icon" href="/file=files/resume.pdf" title="Download Resume" download>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M5 20h14v-2H5v2zm7-18v10l4-4h-3V2h-2v6H8l4 4z"/>
        </svg>
    </a>
</div>

<!-- Info Modal -->
<div id="info_modal">
    <h3>About This Agent</h3>
    <p>This chatbot uses RAG and LLM tech to answer questions about Akshay Abraham's professional background. Ask about skills, experience, or career path!</p>
    <button id="close_modal">Close</button>
</div>

<script>
    document.getElementById('info_icon').onclick = () => {
        document.getElementById('info_modal').style.display = 'block';
    };
    document.getElementById('close_modal').onclick = () => {
        document.getElementById('info_modal').style.display = 'none';
    };
</script>
""")

    # 🤖 Compact Intro Section
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
        <div style='text-align: center; margin-bottom: 15px;'>
        Hello! I'm <strong>Ressy 🤖</strong><br>
        Ask about Akshay's profile
        </div>
        """)

        gr.HTML("""
        <div style="margin: 0 auto;">
            <div class="prompt-row">
                <div class="prompt-container" onclick="fillPromptAndSubmit('What are Akshay\\'s key skills?')">
                    <p>Key skills?</p>
                </div>
                <div class="prompt-container" onclick="fillPromptAndSubmit('Tell me about projects')">
                    <p>Past projects?</p>
                </div>
            </div>
            <div class="prompt-container single-prompt" onclick="fillPromptAndSubmit('What technologies used?')">
                <p>Technologies used?</p>
            </div>
        </div>

        <script>
            function fillPromptAndSubmit(text) {
                const textbox = document.querySelector("#input_textbox textarea");
                textbox.value = text;
                textbox.dispatchEvent(new Event('input', { bubbles: true }));
                const submitButton = document.querySelector("#send_button");
                if (submitButton) {
                    submitButton.click();
                }
            }
        </script>
        """)

    # 💬 Chatbot (initially hidden)
    chatbot = gr.Chatbot(visible=False, elem_id="chatbot")

    # ⌨️ Fixed Position Input
    with gr.Row(elem_id="input_container"):
        msg = gr.Textbox(
            label="",
            placeholder="Ask about Akshay...",
            container=False,
            elem_id="input_textbox",
            lines=1,
            max_lines=3
        )
        submit = gr.Button("➤", elem_id="send_button")

    # 🧠 Response Logic
    def user_submit(message, chat_history):
        chat_history.append({"role": "user", "content": message})
        return "", chat_history, gr.update(visible=False), gr.update(visible=True)

    def bot_reply(chat_history):
        message = chat_history[-1]["content"]
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(
            client, 
            "llama3-70b-8192", 
            message, 
            relevant_excerpts
        )
        chat_history.append({"role": "assistant", "content": bot_message})
        return chat_history

    # 📩 Bind Events
    submit.click(
        user_submit, [msg, chatbot], [msg, chatbot, intro_section, chatbot], show_progress=False
    ).then(
        bot_reply, chatbot, chatbot
    )

    msg.submit(
        user_submit, [msg, chatbot], [msg, chatbot, intro_section, chatbot], show_progress=False
    ).then(
        bot_reply, chatbot, chatbot
    )

    # 🔄 Auto-scroll and intro hiding
    gr.HTML("""
<script>
    const observer = new MutationObserver(() => {
        const bot = document.querySelector("#chatbot");
        if (bot) {
            bot.scrollTo({ top: bot.scrollHeight, behavior: "smooth" });
        }
    });
    observer.observe(document.querySelector("#chatbot"), {
        childList: true,
        subtree: true
    });

    document.querySelector("#input_textbox textarea").addEventListener("keydown", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            document.querySelector("#intro_container").style.display = "none";
            document.querySelector("#chatbot").style.display = "block";
        }
    });
</script>
""")

# 🚀 Launch
if __name__ == "__main__":
    demo.launch()