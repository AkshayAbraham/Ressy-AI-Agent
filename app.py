import gradio as gr
import os
import requests
from groq import Groq
from dotenv import load_dotenv

# Mock implementations for demonstration purposes if actual ones fail
# In a real scenario, these would be your actual functions
def load_text_data(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return "Mock resume data."

def resume_chat_completion(client, model, message, context):
    return f"Mock response based on context: {context}. User query: {message}"

def semantic_search(query, retriever):
    return f"Mock relevant excerpts for: {query}"

def setup_embedding_model(model_name):
    class MockEmbeddingModel:
        def embed_query(self, text):
            return [0.1, 0.2] # Dummy embedding
    return MockEmbeddingModel()

def get_publications():
    return [{"title": "Mock Publication 1", "link": "#"}, {"title": "Mock Publication 2", "link": "#"}]

from langchain.vectorstores import Chroma # Keep import for structural similarity
# Mock Chroma and retriever setup if actual ones fail due to missing libraries
embedding_model = setup_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2")
my_resume = load_text_data("data/resume.txt")
chunks = [chunk.strip() for chunk in my_resume.split("---") if chunk.strip()]
# Use a dummy Chroma object if real one fails
try:
    db = Chroma.from_texts(chunks, embedding_model)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
except Exception as e:
    print(f"Warning: Could not initialize Chroma. Using mock retriever. Error: {e}")
    class MockRetriever:
        def as_retriever(self, search_type, search_kwargs):
            return self
        def get_relevant_documents(self, query):
            return ["Mock relevant document 1", "Mock relevant document 2"]
    retriever = MockRetriever()


# Create cache directory
os.makedirs('.gradio/cached_examples', exist_ok=True)

# --- Setup LLM (Groq) ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY")) # This still requires GROQ_API_KEY env var

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# MODIFIED: send_telegram_message now returns a tuple for updating multiple outputs
def send_telegram_message(message: str):
    """Sends a suggestion message to Telegram and returns a tuple for UI updates."""
    status_message = ""
    # Assume modal_message_output is the element to show success/error
    # And suggestion_section_gradio is the modal itself

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        status_message = "<span style='color: #ff6b6b;'>❌ ERROR: Telegram integration not configured on server.</span>"
        return gr.update(visible=True), status_message # Keep modal open, show error
    
    if not message.strip():
        status_message = "<span style='color: #ff6b6b;'>❌ ERROR: Please enter a message before submitting.</span>"
        return gr.update(visible=True), status_message # Keep modal open, show error

    telegram_message_text = f"🌐 New Contact Message:\n\n{message}"

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": telegram_message_text
        }

        response = requests.post(url, json=payload)
        if response.status_code == 200:
            status_message = "<span style='color: #61dafb;'>✅ Message sent successfully!</span>"
            # For success, hide the modal, show success message
            return gr.update(visible=False), status_message 
        else:
            error_msg = f"Failed to send: {response.status_code} - {response.text}"
            print(f"Telegram API Error: {error_msg}")
            status_message = f"<span style='color: #ff6b6b;'>❌ ERROR: {error_msg}</span>"
            return gr.update(visible=true), status_message # Keep modal open, show error
    except Exception as e:
        error_msg = f"Network error: {str(e)}"
        print(f"Error sending Telegram message: {error_msg}")
        status_message = f"<span style='color: #ff6b6b;'>❌ ERROR: {error_msg}</span>"
        return gr.update(visible=True), status_message # Keep modal open, show error

# --- Custom CSS ---
custom_css = """
body, .gradio-container {
    background-color: #1A1A1A !important;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Hide all loading indicators (important for this solution!) */
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

/* Input container - MADE TRANSPARENT */
#input_container {
    position: relative;
    background-color: transparent !important; /* Make it transparent */
    border: 1px solid #444 !important; /* Keep border or remove if preferred */
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
    min-width: 36px !important; /* Ensure minimum size */
    min-height: 36px !important; /* Ensure minimum size */
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

/* --- TOP ICONS --- */
.top-icons {
    position: fixed !important;
    top: 20px !important;
    right: 20px !important;
    display: flex !important;
    gap: 15px !important;
    z-index: 1000 !important;
}

.top-icons button,
.top-icons a {
    background-color: #333 !important;
    border: none !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    min-height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease, transform 0.2s ease !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

.top-icons button:hover,
.top-icons a:hover {
    background-color: #4a90e2 !important;
    transform: translateY(-2px) !important;
}

.top-icons svg {
    fill: #f0f0f0 !important;
    width: 24px !important;
    height: 24px !important;
    display: block !important;
}
/* --- END TOP ICONS --- */


/* --- INFO MODAL --- */
#info_modal::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    background: transparent !important;
}
#info_modal {
    scrollbar-width: none !important; /* Firefox */
    -ms-overflow-style: none !important; /* IE/Edge */
    
    display: none; /* IMPORTANT: Hide by default */
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #2C2C2C !important; /* Consistent with other modal */
    color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    z-index: 9998; /* Just below the Telegram success animation (9999 or 10000) */
    max-width: 600px; /* Adjust as needed */
    width: 90%; /* Responsive width */
    max-height: 80vh; /* Make it scrollable if content is too long */
    overflow-y: auto; /* Enable scrolling for content */
    box-sizing: border-box; /* Include padding in width/height */
}

#info_modal h3 {
    color: #61dafb;
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 1.5em;
    text-align: center;
}

#info_modal p, #info_modal ul, #info_modal .disclaimer {
    color: #d0d0d0;
    line-height: 1.6;
    margin-bottom: 10px;
}

#info_modal ul {
    padding-left: 20px;
}

#info_modal li {
    margin-bottom: 5px;
}

#info_modal a {
    color: #4a90e2;
    text-decoration: none;
}

#info_modal a:hover {
    text-decoration: underline;
}

#info_modal .disclaimer {
    margin-top: 20px;
    padding: 10px;
    border: 1px solid #444;
    border-radius: 5px;
    background-color: #383838;
    font-size: 0.9em;
}

#info_modal button {
    background-color: #4a90e2; /* Consistent button style */
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 14px;
    margin-top: 20px;
    display: block; /* Make it take full width */
    width: fit-content; /* Adjust to content width */
    margin-left: auto;
    margin-right: auto; /* Center the button */
    transition: background-color 0.2s ease, transform 0.2s ease;
}

#info_modal button:hover {
    background-color: #357ABD;
    transform: scale(1.05);
}
/* --- END INFO MODAL --- */


/* Styles for the Connect With Me modal (Gradio column) */
.gradio-container .suggestion-box {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #2C2C2C !important;
    color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    z-index: 9999; /* Ensure it's on top */
    max-width: 500px;
    width: 90%;
    box-sizing: border-box;
}

.gradio-container .suggestion-box h3 {
    color: #61dafb;
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 1.5em;
    text-align: center;
}

.gradio-container .suggestion-box label {
    display: block;
    margin-bottom: 5px;
    color: #d0d0d0;
}

.gradio-container .suggestion-box textarea {
    width: 100%;
    background-color: #1A1A1A;
    border: 1px solid #444;
    border-radius: 5px;
    padding: 10px;
    color: white;
    box-sizing: border-box;
    margin-bottom: 15px;
    resize: vertical; /* Allow vertical resize */
}

.gradio-container .suggestion-box .button-group {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 15px;
}

.gradio-container .suggestion-box .primary-btn {
    background-color: #4a90e2 !important; /* Force this color */
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 20px !important;
    cursor: pointer !important;
    font-size: 14px !important;
    transition: background-color 0.2s ease, transform 0.2s ease !important;
}

.gradio-container .suggestion-box .primary-btn:hover {
    background-color: #357ABD !important;
    transform: scale(1.05) !important;
}

.gradio-container .suggestion-box .secondary-btn {
    background-color: #555 !important;
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 20px !important;
    cursor: pointer !important;
    font-size: 14px !important;
    transition: background-color 0.2s ease, transform 0.2s ease !important;
}

.gradio-container .suggestion-box .secondary-btn:hover {
    background-color: #666 !important;
    transform: scale(1.05) !important;
}

/* New: Styles for social media icons/links within the modal */
.contact-info {
    margin-bottom: 20px;
    color: #d0d0d0;
    text-align: center;
    line-height: 1.6;
}
.contact-info a {
    color: #4a90e2;
    text-decoration: none;
    font-weight: bold;
}
.contact-info a:hover {
    text-decoration: underline;
}
.social-links {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 20px;
    margin-bottom: 20px;
}
.social-links a {
    display: inline-block;
    transition: transform 0.2s ease;
}
.social-links a:hover {
    transform: translateY(-3px);
}
.social-links svg {
    fill: #61dafb; /* A nice highlight color */
    width: 32px;
    height: 32px;
}
.modal-message-display {
    text-align: center;
    margin-top: 15px;
    font-size: 0.9em;
    min-height: 1.5em; /* Reserve space */
}

/* New: Success/Error Message Display for the Modal */
#modal_python_message_output { /* NEW ID */
    text-align: center;
    margin-top: 15px;
    font-size: 0.9em;
    min-height: 1.5em; /* Reserve space */
    color: white; /* Default color, will be overridden by span styles */
}


/* Remove the success animation modal, as we can't control it with _js */
/* #success_animation_modal {
    display: none;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0,0,0,0.8);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 0 30px rgba(0,0,0,0.6);
    z-index: 10000;
    text-align: center;
    color: white;
    font-size: 1.2em;
}
#success_animation_modal dotlottie-player {
    width: 150px;
    height: 150px;
    margin-bottom: 15px;
} */
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    # MODIFIED: Define a gr.File component to serve the PDF for download
    pdf_file_comp = gr.File(
        value="data/resume.pdf",
        visible=False,
        file_count="single",
        elem_id="pdf_file_component"
    )
    
    # Hidden Gradio Button to act as a bridge from HTML to Python for suggestion
    suggest_trigger_btn = gr.Button(visible=False, elem_id="suggest_trigger_btn_id")

    # Connect With Me Section (This is your Gradio-controlled "modal")
    with gr.Column(visible=False, elem_id="suggestion_section_gradio") as suggestion_section:
        with gr.Column(elem_classes="suggestion-box"):
            gr.Markdown("### Connect with me 👋")
            gr.HTML("""
            <div class="contact-info">
                <p>Feel free to reach out directly or connect on my social channels:</p>
                <p><strong>Email:</strong> <a href="mailto:akshay.abraham@example.com">akshay.abraham@example.com</a></p>
                <p><strong>Phone:</strong> +44 123 456 7890</p>
            </div>
            <div class="social-links">
                <a href="https://www.linkedin.com/in/akshay-abraham/" target="_blank" title="LinkedIn">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-.984-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                    </svg>
                </a>
                <a href="https://github.com/akshay-abraham" target="_blank" title="GitHub">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.087-.731.084-.716.084-.716 1.205.082 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.493.998.108-.776.417-1.305.76-1.605-2.665-.304-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.118-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576c4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                </a>
            </div>
            """)
            suggestion_box = gr.Textbox(
                label="Send a Direct Message:",
                lines=3,
                max_lines=5,
                placeholder="Type your message here, including your contact info (email/phone) if you'd like a response.",
                elem_id="suggestion_box_input_field"
            )
            # NEW: Hidden Textbox to pass message content from JS to Python
            _message_to_send = gr.Textbox(visible=False, elem_id="_message_to_send_input")
            # NEW: Hidden Button that will actually trigger the Python function
            _submit_message_action_btn = gr.Button(visible=False, elem_id="_submit_message_action_btn_id")

            with gr.Row():
                # This button now primarily triggers JavaScript
                suggestion_submit_btn_gradio = gr.Button("Send Message", variant="primary", elem_id="suggestion_submit_btn_gradio_id")
                close_suggestion_btn_gradio = gr.Button("Close")
            
            # MODIFIED: This HTML element is now directly updated by Python for final status
            modal_message_output = gr.HTML('<div class="modal-message-display" id="modal_message_display"></div>', elem_id="modal_python_message_output") 
            # Note: The JS still updates 'modal_message_display' for "Sending...",
            # and 'modal_python_message_output' gets the final Python status.
            # We can merge these two elements into one if we want just one message display.
            # For simplicity, keeping them separate for now, but `modal_message_display` will be overridden.

    # Event handlers for the Gradio components
    def toggle_suggestion_section():
        return gr.update(visible=True)
    
    suggest_trigger_btn.click(
        fn=toggle_suggestion_section,
        outputs=suggestion_section
    )
    
    close_suggestion_btn_gradio.click(
        fn=lambda: gr.update(visible=False), # Hide the modal
        outputs=suggestion_section
    )
    
    # MODIFIED: _submit_message_action_btn now directly updates the modal's visibility
    # and the message output element.
    _submit_message_action_btn.click(
        fn=send_telegram_message,
        inputs=_message_to_send, # Uses the hidden textbox as input
        outputs=[suggestion_section, modal_message_output] # Output to modal visibility and message area
        # show_progress=False is important here to avoid default Gradio loading text
    )

    # REMOVED: The telegram_status_output_bridge.change() with _js is removed due to the TypeError.

    gr.HTML("""
<style>
    /* Your CSS is defined in the custom_css python string above. */
</style>

<div class="top-icons">
    <button id="info_icon" title="About Agent">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M11 9h2V7h-2v2zm0 8h2v-6h-2v6zm1-16C5.48 1 1 5.48 1 11s4.48 10 10 10 10-4.48 10-10S16.52 1 12 1zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
        </svg>
    </button>

    <button id="suggest_icon" title="Connect with me">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17l-.59.59-.58.58V4h16v12zm-9-4h2v2h-2zm0-6h2v4h-2z"/>
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
        Ressy is powered by **cutting-edge RAG (Retrieval-Augmented Generation) and LLM (Large Language Model) technology**. This means I don't just guess; I intelligently search through Akshay's comprehensive resume and use advanced AI to provide you with **accurate, relevant, and insightful answers. 🔍✨**
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

    <button id="close_info_modal">Close</button>
</div>

<script>
    document.addEventListener('DOMContentLoaded', (event) => {
        // Info Modal Logic
        const infoIcon = document.getElementById('info_icon');
        const infoModal = document.getElementById('info_modal');
        const closeInfoModal = document.getElementById('close_info_modal');

        if (infoIcon && infoModal && closeInfoModal) {
            infoIcon.onclick = () => {
                infoModal.style.display = 'block';
            };
            closeInfoModal.onclick = () => {
                infoModal.style.display = 'none';
            };
        }

        // PDF Download Logic
        const pdfFileComponent = document.getElementById('pdf_file_component');
        const customDownloadIcon = document.getElementById('download_icon');

        if (pdfFileComponent && customDownloadIcon) {
            const observer = new MutationObserver((mutations) => {
                for (let mutation of mutations) {
                    if (mutation.type === 'childList' || mutation.type === 'subtree') {
                        const gradioDownloadLink = pdfFileComponent.querySelector('.file-preview a');
                        if (gradioDownloadLink && gradioDownloadLink.href && customDownloadIcon.href === '#') {
                            customDownloadIcon.href = gradioDownloadLink.href;
                            observer.disconnect();
                            break;
                        }
                    }
                }
            });

            observer.observe(pdfFileComponent, { childList: true, subtree: true });

            setTimeout(() => {
                const gradioDownloadLink = pdfFileComponent.querySelector('.file-preview a');
                if (gradioDownloadLink && gradioDownloadLink.href && customDownloadIcon.href === '#') {
                    customDownloadIcon.href = gradioDownloadLink.href;
                }
            }, 1000);
        }

        // SUGGESTION (Connect with me) BUTTON BRIDGE:
        const htmlSuggestIcon = document.getElementById('suggest_icon');
        const gradioSuggestTrigger = document.getElementById('suggest_trigger_btn_id');
        const modalMessageOutput = document.getElementById('modal_python_message_output'); // Get reference to the element that will show Python messages


        if (htmlSuggestIcon && gradioSuggestTrigger) {
            htmlSuggestIcon.onclick = () => {
                gradioSuggestTrigger.click(); // Programmatically click the hidden Gradio button
                // Clear any previous messages when opening the modal
                if(modalMessageOutput) modalMessageOutput.innerHTML = ''; 
                const suggestionInput = document.getElementById('suggestion_box_input_field').querySelector('textarea');
                if(suggestionInput) suggestionInput.value = ''; // Clear input field
            };
        }

        // Handle the visible "Send Message" button click
        const visibleSubmitBtn = document.getElementById('suggestion_submit_btn_gradio_id');
        const suggestionInput = document.getElementById('suggestion_box_input_field').querySelector('textarea');
        const hiddenMessageInput = document.getElementById('_message_to_send_input').querySelector('textarea');
        const hiddenSubmitActionBtn = document.getElementById('_submit_message_action_btn_id');
        const modalMessageDisplay = document.getElementById('modal_message_display'); // This is the temporary "Sending..." message

        if (visibleSubmitBtn && suggestionInput && hiddenMessageInput && hiddenSubmitActionBtn && modalMessageDisplay) {
            visibleSubmitBtn.onclick = () => {
                const message = suggestionInput.value;
                
                // 1. Clear input field immediately
                suggestionInput.value = '';
                
                // 2. Show "Sending..." message (temporary)
                modalMessageDisplay.style.color = '#d0d0d0'; // Reset color
                modalMessageDisplay.textContent = 'Sending...';
                
                // Clear the long-term message output in case of multiple sends
                if (modalMessageOutput) {
                    modalMessageOutput.innerHTML = '';
                }

                // 3. Pass the message to the hidden input Gradio component
                hiddenMessageInput.value = message;
                hiddenMessageInput.dispatchEvent(new Event('input', { bubbles: true }));

                // 4. Programmatically click the hidden Gradio button to trigger the Python function
                hiddenSubmitActionBtn.click();
            };
        }
    }); // End DOMContentLoaded
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