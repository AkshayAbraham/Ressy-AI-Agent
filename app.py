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

# --- Custom CSS with Cute Robot Animations ---
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

/* Animated robot container */
#robot_container {
    width: 150px;
    height: 150px;
    margin: 0 auto 20px auto;
    position: relative;
    animation: float 4s ease-in-out infinite;
}

/* Floating animation */
@keyframes float {
    0%, 100% { transform: translateY(0) rotate(-5deg); }
    50% { transform: translateY(-15px) rotate(5deg); }
}

/* Robot antenna animation */
@keyframes antenna {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

/* Robot eye animation */
@keyframes blink {
    0%, 45%, 55%, 100% { transform: scaleY(1); }
    50% { transform: scaleY(0.1); }
}

/* Chatbot container */
#chatbot {
    background-color: #1A1A1A !important;
    border: none !important;
    padding: 0 !important;
    transition: opacity 0.3s ease;
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

# SVG for cute animated robot
animated_robot_svg = """
<div id="robot_container">
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <style>
        /* Robot body */
        .robot-body {
            fill: #4a90e2;
            stroke: #357ABD;
            stroke-width: 3;
        }
        /* Robot head */
        .robot-head {
            fill: #4a90e2;
            stroke: #357ABD;
            stroke-width: 3;
        }
        /* Antenna */
        .antenna {
            stroke: #FFD700;
            stroke-width: 4;
            stroke-linecap: round;
            animation: antenna 2s ease-in-out infinite;
        }
        .antenna-ball {
            fill: #FFD700;
            animation: antenna 2s ease-in-out infinite;
        }
        /* Eyes */
        .eye {
            fill: white;
        }
        .pupil {
            fill: #1A1A1A;
            animation: blink 3s infinite;
        }
        /* Mouth */
        .mouth {
            fill: none;
            stroke: white;
            stroke-width: 3;
            stroke-linecap: round;
        }
        /* Arms */
        .arm {
            fill: #357ABD;
            stroke: #2a5a8f;
            stroke-width: 2;
        }
    </style>
    
    <!-- Robot Body -->
    <rect class="robot-body" x="50" y="90" width="100" height="80" rx="10"/>
    
    <!-- Robot Head -->
    <circle class="robot-head" cx="100" cy="70" r="30"/>
    
    <!-- Antenna -->
    <line class="antenna" x1="100" y1="40" x2="85" x2="85" y2="10"/>
    <circle class="antenna-ball" cx="85" cy="10" r="6"/>
    
    <!-- Eyes -->
    <circle class="eye" cx="85" cy="65" r="8"/>
    <circle class="eye" cx="115" cy="65" r="8"/>
    <circle class="pupil" cx="85" cy="65" r="3"/>
    <circle class="pupil" cx="115" cy="65" r="3"/>
    
    <!-- Mouth -->
    <path class="mouth" d="M85 85 Q100 95 115 85"/>
    
    <!-- Arms -->
    <rect class="arm" x="30" y="100" width="20" height="15" rx="5"/>
    <rect class="arm" x="150" y="100" width="20" height="15" rx="5"/>
</svg>
</div>
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")

    # 🤖 Animated Robot Intro
    with gr.Column(visible=True, elem_id="intro_container") as intro_section:
        gr.HTML(animated_robot_svg)
        gr.Markdown("""
        <div style='animation: fadeIn 0.8s ease-out;'>
        Hello! I'm Robo, your resume assistant 🤖<br>
        Ask me anything about Akshay's professional journey!
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