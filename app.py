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
    width: 150px !important;
    height: 150px !important;
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

/* Generic fallback to hide any floating scroll-to-bottom button */
button[aria-label="Scroll to bottom"] {
    display: none !important;
}

"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
<style>
    .top-icons {
        display: flex;
        justify-content: flex-end;
        gap: 25px;
        padding: 10px 20px;
    }

    .top-icons button,
    .top-icons a {
        background: none;
        border: none;
        cursor: pointer;
        transition: transform 0.2s ease;
    }

    .top-icons button:hover svg,
    .top-icons a:hover svg {
        transform: scale(1.2);
    }

    .top-icons svg {
        fill: #ffffff;
        width: 28px;
        height: 28px;
        transition: transform 0.2s ease;
    }

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

    #info_modal button {
        margin-top: 10px;
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
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

</style>

<div class="top-icons">
    <!-- Info Icon -->
    <button id="info_icon" title="About Agent">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M11 9h2V7h-2v2zm0 8h2v-6h-2v6zm1-16C5.48 1 1 5.48 1 11s4.48 10 10 10 10-4.48 10-10S16.52 1 12 1zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
        </svg>
    </button>

    <!-- Download Icon -->
    <a id="download_icon" href="/file=files/resume.pdf" title="Download Resume" download>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M5 20h14v-2H5v2zm7-18v10l4-4h-3V2h-2v6H8l4 4z"/>
        </svg>
    </a>
</div>

<!-- Info Modal -->
<div id="info_modal">
    <h3>About This Agent</h3>
    <p>This chatbot uses RAG and LLM tech to answer questions about Akshay Abraham’s professional background. Ask about skills, experience, or career path!</p>
    <button onclick="document.getElementById('info_modal').style.display='none'">Close</button>
</div>

<script>
    // Show info modal on icon click
    document.getElementById('info_icon').onclick = () => {
        document.getElementById('info_modal').style.display = 'block';
    };
</script>
""")



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

# --- Respond logic split in two parts ---
    def user_submit(message, chat_history):
        chat_history.append({"role": "user", "content": message})
        return "", chat_history, gr.update(visible=False), gr.update(visible=True)

    def bot_reply(chat_history):
        message = chat_history[-1]["content"]
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(
            client, 
            "llama-3.3-70b-versatile", 
            message, 
            relevant_excerpts
        )
        chat_history.append({"role": "assistant", "content": bot_message})
        return chat_history

    # 📩 Bind interactions
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

    # 🧠 JS for UI behavior
    gr.HTML("""
<script>
    // Smooth auto-scroll to bottom of chatbot
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

    // Clear textbox on send (button or Enter)
    const textbox = document.querySelector("#input_textbox textarea");
    const button = document.querySelector("#send_button");

    function clearAndHideIntro() {
        textbox.value = "";
        const intro = document.querySelector("#intro_container");
        const chatbot = document.querySelector("#chatbot");
        if (intro) intro.style.display = "none";
        if (chatbot) chatbot.style.display = "block";
    }

    // Clear on button click
    button.addEventListener("click", clearAndHideIntro);

    // Clear on Enter key
    textbox.addEventListener("keydown", function(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            setTimeout(clearAndHideIntro, 10);
        }
    });
</script>
""")

# 🚀 Launch
if __name__ == "__main__":
    demo.launch()