import gradio as gr
from langchain.vectorstores import Chroma
from utils import (
    load_text_data,
    resume_chat_completion,
    semantic_search,
    setup_embedding_model,
    get_publications
)
import os
from groq import Groq
from dotenv import load_dotenv

# For Telegram Sending (Only this one remains)
from telegram import Bot
from telegram.error import TelegramError

# Create cache directory
os.makedirs('.gradio/cached_examples', exist_ok=True)

# --- Load Models & Data ---
embedding_model = setup_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2")
my_resume = load_text_data("data/resume.txt")
chunks = [chunk.strip() for chunk in my_resume.split("---") if chunk.strip()]
db = Chroma.from_texts(chunks, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --- Setup LLM (Groq) ---
load_dotenv() # Load environment variables from .env file (for local testing)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Telegram Configuration ---
# These will be pulled from Hugging Face Secret Variables in deployment
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

telegram_bot = None
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    try:
        telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        print("Telegram bot initialized successfully.")
    except Exception as e:
        print(f"Error initializing Telegram bot: {e}")
        telegram_bot = None
else:
    print("WARNING: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID environment variables are not set. Telegram sending functionality will not work.")
    print("Please set them in your .env file locally, or as Secret Variables in Hugging Face Spaces.")

# --- Telegram Sending Function ---
async def send_message_telegram(sender_name, message_content):
    if not telegram_bot or not TELEGRAM_CHAT_ID:
        return "Telegram sending is not configured on the server. Please check server logs for details."

    telegram_message = (
        f"New Message from Ressy AI Resume Chatbot:\n\n"
        f"Sender Name: {sender_name}\n"
        f"Message:\n{message_content}"
    )

    try:
        # send_message is an async function, so it needs to be awaited
        await telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=telegram_message)
        return "Message sent successfully via Telegram!"
    except TelegramError as e:
        print(f"Telegram API Error: {e}")
        return f"Failed to send message via Telegram. Error: {e}"
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return f"Failed to send message via Telegram. Unknown error: {e}"

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

/* Hide Scrollbars (for vertical on chatbot, and horizontal on prompt container) */
#chatbot::-webkit-scrollbar, #horizontal-prompts::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important; /* Hide horizontal scrollbar */
    background: transparent !important;
}
#chatbot, #horizontal-prompts {
    scrollbar-width: none !important; /* Firefox */
    -ms-overflow-style: none !important; /* IE/Edge */
}

/* Lottie container styling */
#lottie_container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    margin: 0 auto;
    width: 100px !important;
    height: 100px !important;
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
    overflow-y: auto !important; /* Maintain vertical scroll functionality */
}

/* Intro section */
#intro_container {
    text-align: center;
    margin: 20px auto;
    max-width: 500px;
    animation: fadeIn 0.8s ease-out;
}

/* New: Horizontal Prompts Container */
#horizontal-prompts {
    display: flex;
    gap: 20px; /* Space between prompt containers */
    overflow-x: auto; /* Enable horizontal scrolling */
    padding: 10px 0; /* Add some padding if needed, not affecting scrollbar */
    justify-content: flex-start; /* Align items to the start, allowing overflow */
    margin: 10px auto 0 auto; /* DECREASED THIS MARGIN-TOP to 10px */
    max-width: 100%; /* Ensure it can take full width */
    white-space: nowrap; /* Prevent items from wrapping */
    scroll-snap-type: x mandatory; /* Optional: for smoother snapping to items */
}

.prompt-container {
    background-color: #282828;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    border: 1px solid #3a3a3a;
    text-align: center;
    min-width: 180px; /* Ensures minimum width for each card */
    flex-shrink: 0; /* Prevent items from shrinking */
    transition: all 0.3s ease;
    cursor: pointer;
    display: flex; /* For centering content within the card */
    align-items: center; /* Vertically center content */
    justify-content: center; /* Horizontally center content */
    scroll-snap-align: center; /* Optional: for smoother snapping to items */
}

.prompt-container:hover {
    background-color: #383838;
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}

.prompt-container p {
    margin: 0;
    color: #ffffff;
    font-size: 1em;
    line-height: 1.4;
    white-space: normal; /* Allow text inside cards to wrap */
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
    margin: 20px auto; /* This margin controls the space above the input box */
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
    resize: vertical;
    min-height: 80px;
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

/* Styles for the info modal scrollbar */
#info_modal::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    background: transparent !important;
}
#info_modal {
    scrollbar-width: none !important; /* Firefox */
    -ms-overflow-style: none !important; /* IE/Edge */
}

/* NEW: Connect Modal Styles */
#connect_modal {
    display: none;
    position: fixed;
    top: 10%;
    left: 50%;
    transform: translateX(-50%);
    background: #2C2C2C;
    color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    z-index: 9999;
    max-width: 600px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    box-sizing: border-box;
}

#connect_modal h3 {
    color: #61dafb;
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 1.5em;
    text-align: center;
}

#connect_modal p {
    font-size: 1em;
    line-height: 1.6;
    margin-bottom: 20px;
    color: #d0d0d0;
    text-align: center;
}

#connect_modal .contact-link {
    color: #4a90e2;
    text-decoration: none;
    font-weight: bold;
}

#connect_modal .contact-link:hover {
    text-decoration: underline;
}

#connect_modal .message-form-section {
    margin-top: 25px;
    padding-top: 20px;
    border-top: 1px dashed #444;
}

#connect_modal .message-form-section h4 {
    text-align: center;
    margin-bottom: 15px;
    color: #fff;
    font-size: 1.2em;
}

#connect_modal .message-form-section label {
    display: block;
    margin-bottom: 5px;
    color: #d0d0d0;
    font-size: 0.9em;
}

#connect_modal .message-form-section input[type="text"],
#connect_modal .message-form-section input[type="email"], /* Keeping email type for robustness if needed, but not used in form */
#connect_modal .message-form-section textarea {
    width: calc(100% - 20px);
    padding: 10px;
    margin-bottom: 15px;
    border: 1px solid #555;
    border-radius: 5px;
    background-color: #3A3A3A;
    color: white;
    font-size: 1em;
}

#connect_modal .message-form-section textarea {
    resize: vertical;
    min-height: 80px;
}

#connect_modal .message-form-section button {
    background-color: #4a90e2;
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 25px;
    font-size: 1em;
    cursor: pointer;
    transition: background-color 0.2s ease, transform 0.2s ease;
    display: block;
    width: 100%;
    margin-bottom: 10px;
}

#connect_modal .message-form-section button:hover {
    background-color: #357ABD;
    transform: scale(1.02);
}

#connect_modal .status-message {
    margin-top: 10px;
    text-align: center;
    font-size: 0.9em;
    color: #d0d0d0;
}

#connect_modal button#close_connect_modal {
    margin-top: 25px;
    background-color: #666;
    color: white;
    border: none;
    padding: 10px 16px;
    border-radius: 20px;
    font-size: 14px;
    cursor: pointer;
    transition: background-color 0.2s ease, transform 0.2s ease;
    display: block;
    width: 100%;
}
#connect_modal button#close_connect_modal:hover {
    background-color: #555;
    transform: scale(1.02);
}

"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    pdf_file_comp = gr.File(
        value="data/resume.pdf",
        visible=False,
        file_count="single",
        elem_id="pdf_file_component"
    )

    # All single '{' and '}' within the JavaScript part of this f-string must be escaped to '{{' and '}}'
    gr.HTML(f"""
<style>
    /* Ensure modals are hidden by default */
    #info_modal, #connect_modal {{ display: none; }}
</style>

<div class="top-icons">
    <button id="info_icon" title="About Agent">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M11 9h2V7h-2v2zm0 8h2v-6h-2v6zm1-16C5.48 1 1 5.48 1 11s4.48 10 10 10 10-4.48 10-10S16.52 1 12 1zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
        </svg>
    </button>

    <button id="connect_icon" title="Connect with Akshay">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M12 12c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-10c-5.52 0-10 4.48-10 10s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-2.76 0-5 2.24-5 5h2c0-1.66 1.34-3 3-3s3 1.34 3 3h2c0-2.76-2.24-5-5-5z"/>
        </svg>
    </button>

    <a id="download_icon" href="#" title="Download Resume" download="Akshay_Abraham_Resume.pdf">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M5 20h14v-2H5v2zm7-18v10l4-4h-3V2h-2v6H8l4 4z"/>
        </svg>
    </a>
</div>

<div id="info_modal">
    <h3>About Your AI Resume Assistant: Ressy 🤖</h3>
    <p>
        Welcome to Ressy, your intelligent guide to Akshay Abraham's professional journey! 👋
        <br><br>
        Ressy is powered by **cutting-edge RAG (Retrieval-Augmented Generation) and LLM (Large Language Model) technology**. This means I don't just guess; I intelligently search through Akshay's comprehensive resume and use advanced AI to provide you with **accurate, relevant, and insightful answers**. 🔍✨
        <br><br>
        What can I help you discover? 💡
        <ul>
            <li>Skills Deep Dive: Uncover Akshay's expertise in areas like **Machine Learning 🧠, Deep Learning 📊, Data Analysis 📈, Python 🐍, SQL 🗄️, cloud platforms (AWS) ☁️**, and various tools and frameworks.</li>
            <li>Project Showcase: Explore detailed information about his impactful projects, including **customer churn prediction 📉** and **NLP-based sentiment analysis 💬**.</li>
            <li>Experience & Impact: Learn about his professional roles, responsibilities, and the tangible results he delivered. 🚀</li>
            <li>Career Trajectory: Understand his career path and future aspirations. 🌟</li>
        </ul>
        Think of me as your dedicated, instant, and interactive resume reader. I'm here to streamline your search for information, making it easier to see how Akshay's background aligns with your needs. 🎯
        <br><br>
        Ready to start exploring? Ask away! 💬
    </p>

    <div class="disclaimer">
        ⚠️ **Important Note:** As an AI, I may occasionally make mistakes or misinterpret context. For the most accurate and up-to-date information, or to connect directly, please refer to Akshay Abraham's official LinkedIn profile: <a href="https://www.linkedin.com/in/akshay-abraham/" target="_blank" rel="noopener noreferrer">Connect with Akshay on LinkedIn 🔗</a>
    </div>

    <button id="close_modal">Close</button>
</div>

<div id="connect_modal">
    <h3>Connect with Akshay 🤝</h3>
    <p>Feel free to reach out using the details below or send a message directly via Telegram.</p>

    <p>
        <strong>Email:</strong> <a href="mailto:akshayabraham02@gmail.com" class="contact-link">akshayabraham02@gmail.com</a><br>
        <strong>Phone:</strong> <a href="tel:+44781815774" class="contact-link">+44781815774</a><br>
        <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/akshay-abraham" target="_blank" rel="noopener noreferrer" class="contact-link">linkedin.com/in/akshay-abraham</a>
    </p>

    <div class="message-form-section">
        <h4>Send via Telegram:</h4>
        <label for="telegram_sender_name_input">Your Name:</label>
        <input type="text" id="telegram_sender_name_input" placeholder="Your Name" />

        <label for="telegram_message_content_input">Message:</label>
        <textarea id="telegram_message_content_input" placeholder="Type your message here..."></textarea>

        <button id="send_telegram_button">Send Telegram Message</button>
        <p id="telegram_status_message" class="status-message"></p>
    </div>

    <button id="close_connect_modal">Close</button>
</div>


<script>
    // Show info modal on icon click
    document.getElementById('info_icon').onclick = () => {{ /* Escaped */
        document.getElementById('info_modal').style.display = 'block';
    }}; /* Escaped */
    document.getElementById('close_modal').onclick = () => {{ /* Escaped */
        document.getElementById('info_modal').style.display = 'none';
    }}; /* Escaped */

    // NEW: Show Connect modal on icon click
    document.getElementById('connect_icon').onclick = () => {{ /* Escaped */
        document.getElementById('connect_modal').style.display = 'block';
        // Clear status message on open
        document.getElementById('telegram_status_message').textContent = '';
    }}; /* Escaped */
    document.getElementById('close_connect_modal').onclick = () => {{ /* Escaped */
        document.getElementById('connect_modal').style.display = 'none';
        document.getElementById('telegram_status_message').textContent = '';
    }}; /* Escaped */

    // MODIFIED: Script to get Gradio File URL for download and update the link
    document.addEventListener('DOMContentLoaded', (event) => {{ /* Escaped */
        setTimeout(() => {{ /* Escaped */
            const fileContainer = document.getElementById('pdf_file_component');
            if (fileContainer) {{ /* Escaped */
                const gradioDownloadLink = fileContainer.querySelector('.file-preview a');
                const customDownloadIcon = document.getElementById('download_icon');

                if (gradioDownloadLink && customDownloadIcon) {{ /* Escaped */
                    customDownloadIcon.href = gradioDownloadLink.href;
                }} /* Escaped */
            }} /* Escaped */
        }}, 500);
    }}); /* Escaped */

    // NEW: JavaScript for sending Telegram message via Gradio backend
    document.getElementById('send_telegram_button').onclick = async () => {{ /* Escaped */
        const senderName = document.getElementById('telegram_sender_name_input').value;
        const messageContent = document.getElementById('telegram_message_content_input').value;
        const statusMessage = document.getElementById('telegram_status_message');

        if (!senderName || !messageContent) {{ /* Escaped */
            statusMessage.style.color = '#ff6b6b';
            statusMessage.textContent = 'Please fill in all Telegram fields.';
            return;
        }} /* Escaped */

        statusMessage.style.color = '#d0d0d0';
        statusMessage.textContent = 'Sending Telegram message...';

        const response = await fetch('/run/send_message_telegram', {{ /* Escaped */
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }}, /* Escaped */
            body: JSON.stringify({{ data: [senderName, messageContent] }}) /* Escaped */
        }}); /* Escaped */

        const result = await response.json();
        if (result.data && result.data.length > 0) {{ /* Escaped */
            statusMessage.textContent = result.data[0];
            if (result.data[0].includes('successfully')) {{ /* Escaped */
                statusMessage.style.color = '#6bff6b';
                document.getElementById('telegram_sender_name_input').value = '';
                document.getElementById('telegram_message_content_input').value = '';
            }} else {{ statusMessage.style.color = '#ff6b6b'; }} /* Escaped */
        }} else {{ /* Escaped */
            statusMessage.style.color = '#ff6b6b';
            statusMessage.textContent = 'Unknown error sending Telegram message.';
        }} /* Escaped */
    }}; /* Escaped */

</script>
""")

    # 🤖 Intro Section with Lottie + Example Prompts
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
        <div style='animation: fadeIn 0.8s ease-out; text-align: center;'>
        Hello! I'm your AI assistant <strong>Ressy 🤖</strong><br>
        Ready to explore Akshay's professional background!
        </div>
        """)

        gr.HTML("""
        <div id="horizontal-prompts">
            <div class="prompt-container" onclick="fillPromptAndSubmit('What are Akshay\\'s key skills?')">
                <p>What are Akshay's key skills?</p>
            </div>
            <div class="prompt-container" onclick="fillPromptAndSubmit('Tell me about Akshay\\'s past projects.')">
                <p>Tell me about Akshay's past projects.</p>
            </div>
            <div class="prompt-container" onclick="fillPromptAndSubmit('What tools or frameworks has Akshay used?')">
                <p>What tools or frameworks has Akshay used?</p>
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

    # 💬 Chatbot
    chatbot = gr.Chatbot(visible=False, type="messages", height=400, elem_id="chatbot")

    # ⌨️ Input Area
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

    # 🧠 Response Logic
    def user_submit(message, chat_history):
        chat_history.append({"role": "user", "content": message})
        return "", chat_history, gr.update(visible=False), gr.update(visible=True)

    def bot_reply(chat_history):
        message = chat_history[-1]["content"]
        relevant_excerpts = semantic_search(message, retriever)

        # Check if the question is about publications/research
        if any(keyword in message.lower() for keyword in [
                            "publication", "publications", "published",
                            "research", "researches",
                            "paper", "papers",
                            "article", "articles",
                            "journal", "journals",
                            "author", "authored",
                            "contribution", "contributions",
                            "cite", "citations"
                        ]):
            publications = get_publications()
            publications_text = "\n".join(
                [f"- {pub['title']} ({pub['link']})" for pub in publications]
            )
            relevant_excerpts += f"\n\nAdditional Publications:\n{publications_text}"

        bot_message = resume_chat_completion(
            client,
            "llama-3.3-70b-versatile",
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

    # WORKAROUND for older Gradio versions: Expose send_message_telegram via a hidden component
    # This creates the /run/send_message_telegram endpoint
    hidden_telegram_inputs_name = gr.Textbox(visible=False)
    hidden_telegram_inputs_message = gr.Textbox(visible=False)
    hidden_telegram_outputs_status = gr.Textbox(visible=False)

    # This hidden button's click event registers the function as a Gradio API endpoint.
    # The `api_name` parameter ensures it's exposed under the name expected by your JavaScript.
    hidden_telegram_trigger_button = gr.Button(
        value="Hidden Telegram Trigger", # This value doesn't matter as the button is hidden
        visible=False
    )
    hidden_telegram_trigger_button.click(
        fn=send_message_telegram,
        inputs=[hidden_telegram_inputs_name, hidden_telegram_inputs_message],
        outputs=[hidden_telegram_outputs_status],
        api_name="send_message_telegram" # This is CRITICAL for your JavaScript fetch call
    )


    # 🔄 Scroll + Hide Intro
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

    const textbox = document.querySelector("#input_textbox textarea");
    const button = document.querySelector("#send_button");

    function clearAndHideIntro() {
        textbox.value = "";
        const intro = document.querySelector("#intro_container");
        const chatbot = document.querySelector("#chatbot");
        if (intro) intro.style.display = "none";
        if (chatbot) chatbot.style.display = "block";
    }

    button.addEventListener("click", clearAndHideIntro);
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