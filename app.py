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

# For Email Sending
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# For Telegram Sending (ADD THIS IMPORT)
from telegram import Bot
from telegram.error import TelegramError

# Create cache directory
os.makedirs('.gradio/cached_examples', exist_ok=True)

# --- Load Models & Data ---
embedding_model = setup_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2")
my_resume_chunks = load_text_data("data/resume.txt") # load_text_data now returns chunks
db = Chroma.from_texts(my_resume_chunks, embedding_model) # Use the chunks directly
# Improved retriever configuration (as discussed previously for publications)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

# --- Setup LLM (Groq) ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Email Configuration ---
SENDER_EMAIL = os.getenv("SENDER_EMAIL") # Your email address
SENDER_EMAIL_PASSWORD = os.getenv("SENDER_EMAIL_PASSWORD") # Your email password or app password
RECEIVER_EMAIL = SENDER_EMAIL # Send to yourself

if not SENDER_EMAIL or not SENDER_EMAIL_PASSWORD:
    print("WARNING: SENDER_EMAIL and/or SENDER_EMAIL_PASSWORD environment variables are not set. Email sending functionality will not work.")
    print("Please set them in your .env file: SENDER_EMAIL=your_email@example.com, SENDER_EMAIL_PASSWORD=your_app_password")

# --- Telegram Configuration (ADD THIS SECTION) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

telegram_bot = None
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    try:
        telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        print("Telegram bot initialized.")
    except Exception as e:
        print(f"Error initializing Telegram bot: {e}")
        telegram_bot = None
else:
    print("WARNING: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID environment variables are not set. Telegram sending functionality will not work.")
    print("Please set them in your .env file.")

# --- Email Sending Function ---
def send_message_email(sender_name, sender_email, message_content):
    if not SENDER_EMAIL or not SENDER_EMAIL_PASSWORD:
        return "Email sending is not configured on the server."

    msg = MIMEMultipart()
    msg['From'] = f"{sender_name} <{sender_email}>"
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"Message from Ressy AI Resume Chatbot - {sender_name}"

    body = f"""
    You've received a message from your AI Resume Chatbot:

    Sender Name: {sender_name}
    Sender Email: {sender_email}
    Message:
    {message_content}

    ---
    This email was sent via your Gradio AI Resume Chatbot.
    """
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(SENDER_EMAIL, SENDER_EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
        server.quit()
        return "Message sent successfully!"
    except Exception as e:
        print(f"Error sending email: {e}")
        return f"Failed to send message via email. Error: {e}"

# --- Telegram Sending Function (ADD THIS FUNCTION) ---
async def send_message_telegram(sender_name, message_content):
    if not telegram_bot or not TELEGRAM_CHAT_ID:
        return "Telegram sending is not configured on the server."

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


# --- Custom CSS (Ensure your CSS is properly formatted and applied) ---
# Add new CSS rules for the Telegram section in your Connect Modal
# For brevity, I'm just showing the additions/modifications to the existing CSS.
# You will integrate these into your 'custom_css' variable.
custom_css = """
/* ... (your existing custom_css) ... */

/* New: Connect Modal Styles */
#connect_modal {
    /* ... (your existing connect_modal styles) ... */
    max-width: 600px; /* Slightly wider to accommodate Telegram */
}

#connect_modal .message-form-section { /* New wrapper for form sections */
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

#connect_modal .message-form-section input[type="text"],
#connect_modal .message-form-section input[type="email"],
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
    margin-bottom: 10px; /* Space between buttons */
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
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    pdf_file_comp = gr.File(
        value="data/resume.pdf",
        visible=False,
        file_count="single",
        elem_id="pdf_file_component"
    )

    gr.HTML(f"""
<style>
    /* Ensure modals are hidden by default */
    #info_modal, #connect_modal {{ display: none; }}
    /* ... (your existing top-icons and modal styles here) ... */
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
    <button id="close_modal">Close</button>
</div>

<div id="connect_modal">
    <h3>Connect with Akshay 🤝</h3>
    <p>Feel free to reach out using the details below or send a message directly.</p>

    <p>
        <strong>Email:</strong> <a href="mailto:akshayabraham02@gmail.com" class="contact-link">akshayabraham02@gmail.com</a><br>
        <strong>Phone:</strong> <a href="tel:+44781815774" class="contact-link">+44781815774</a><br>
        <strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/akshay-abraham" target="_blank" rel="noopener noreferrer" class="contact-link">linkedin.com/in/akshay-abraham</a>
    </p>

    <div class="message-form-section">
        <h4>Send via Email:</h4>
        <label for="email_sender_name_input">Your Name:</label>
        <input type="text" id="email_sender_name_input" placeholder="Your Name" />

        <label for="email_sender_email_input">Your Email:</label>
        <input type="email" id="email_sender_email_input" placeholder="your_email@example.com" />

        <label for="email_message_content_input">Message:</label>
        <textarea id="email_message_content_input" placeholder="Type your message here..."></textarea>

        <button id="send_email_button">Send Email</button>
        <p id="email_status_message" class="status-message"></p>
    </div>

    <div class="message-form-section" style="margin-top: 25px;">
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
    document.getElementById('info_icon').onclick = () => {
        document.getElementById('info_modal').style.display = 'block';
    };
    document.getElementById('close_modal').onclick = () => {
        document.getElementById('info_modal').style.display = 'none';
    };

    // NEW: Show Connect modal on icon click
    document.getElementById('connect_icon').onclick = () => {
        document.getElementById('connect_modal').style.display = 'block';
        // Clear status messages on open
        document.getElementById('email_status_message').textContent = '';
        document.getElementById('telegram_status_message').textContent = '';
    };
    document.getElementById('close_connect_modal').onclick = () => {
        document.getElementById('connect_modal').style.display = 'none';
        document.getElementById('email_status_message').textContent = '';
        document.getElementById('telegram_status_message').textContent = '';
    };

    // MODIFIED: Script to get Gradio File URL for download and update the link
    document.addEventListener('DOMContentLoaded', (event) => {
        setTimeout(() => {
            const fileContainer = document.getElementById('pdf_file_component');
            if (fileContainer) {
                const gradioDownloadLink = fileContainer.querySelector('.file-preview a');
                const customDownloadIcon = document.getElementById('download_icon');

                if (gradioDownloadLink && customDownloadIcon) {
                    customDownloadIcon.href = gradioDownloadLink.href;
                }
            }
        }, 500);
    });

    // NEW: JavaScript for sending email via Gradio backend
    document.getElementById('send_email_button').onclick = async () => {
        const senderName = document.getElementById('email_sender_name_input').value;
        const senderEmail = document.getElementById('email_sender_email_input').value;
        const messageContent = document.getElementById('email_message_content_input').value;
        const statusMessage = document.getElementById('email_status_message');

        if (!senderName || !senderEmail || !messageContent) {
            statusMessage.style.color = '#ff6b6b';
            statusMessage.textContent = 'Please fill in all email fields.';
            return;
        }
        if (!senderEmail.includes('@') || !senderEmail.includes('.')) {
            statusMessage.style.color = '#ff6b6b';
            statusMessage.textContent = 'Please enter a valid email address.';
            return;
        }

        statusMessage.style.color = '#d0d0d0';
        statusMessage.textContent = 'Sending email...';

        const response = await fetch('/run/send_message_email', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: [senderName, senderEmail, messageContent] })
        });

        const result = await response.json();
        if (result.data && result.data.length > 0) {
            statusMessage.textContent = result.data[0];
            if (result.data[0].includes('successfully')) {
                statusMessage.style.color = '#6bff6b';
                document.getElementById('email_sender_name_input').value = '';
                document.getElementById('email_sender_email_input').value = '';
                document.getElementById('email_message_content_input').value = '';
            } else { statusMessage.style.color = '#ff6b6b'; }
        } else {
            statusMessage.style.color = '#ff6b6b';
            statusMessage.textContent = 'Unknown error sending email.';
        }
    };

    // NEW: JavaScript for sending Telegram message via Gradio backend
    document.getElementById('send_telegram_button').onclick = async () => {
        const senderName = document.getElementById('telegram_sender_name_input').value;
        const messageContent = document.getElementById('telegram_message_content_input').value;
        const statusMessage = document.getElementById('telegram_status_message');

        if (!senderName || !messageContent) {
            statusMessage.style.color = '#ff6b6b';
            statusMessage.textContent = 'Please fill in all Telegram fields.';
            return;
        }

        statusMessage.style.color = '#d0d0d0';
        statusMessage.textContent = 'Sending Telegram message...';

        const response = await fetch('/run/send_message_telegram', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: [senderName, messageContent] })
        });

        const result = await response.json();
        if (result.data && result.data.length > 0) {
            statusMessage.textContent = result.data[0];
            if (result.data[0].includes('successfully')) {
                statusMessage.style.color = '#6bff6b';
                document.getElementById('telegram_sender_name_input').value = '';
                document.getElementById('telegram_message_content_input').value = '';
            } else { statusMessage.style.color = '#ff6b6b'; }
        } else {
            statusMessage.style.color = '#ff6b6b';
            statusMessage.textContent = 'Unknown error sending Telegram message.';
        }
    };

</script>
""")

    # 🤖 Intro Section with Lottie + Example Prompts
    with gr.Column(visible=True, elem_id="intro_container") as intro_section:
        # ... (your existing intro section HTML, Markdown, and example prompts) ...
        pass # Placeholder for brevity, keep your existing content here

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

    # NEW: Bind Telegram sending function to a Gradio API endpoint
    demo.api_endpoints.append(
        gr.APIRoute(
            path="/run/send_message_telegram",
            methods=["POST"],
            endpoint=send_message_telegram,
            inputs=[gr.Textbox(), gr.Textbox()], # sender_name, message_content
            outputs=[gr.Textbox()] # status message
        )
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