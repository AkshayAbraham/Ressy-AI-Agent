import gradio as gr
from langchain.vectorstores import Chroma
from utils import (
    load_text_data,
    resume_chat_completion,
    semantic_search,
    setup_embedding_model,
    get_publications  # Add this import
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
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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
"""

# --- Gradio UI ---
with gr.Blocks(css=custom_css) as demo:
    # MODIFIED: Define a gr.File component to serve the PDF for download
    # It's hidden, but Gradio will generate a downloadable URL for it.
    # The 'file_count="single"' and 'elem_id' are important.
    pdf_file_comp = gr.File(
        value="data/resume.pdf",
        visible=False, # Keep hidden
        file_count="single",
        elem_id="pdf_file_component" # This ID helps us target it with JS
    )

    gr.HTML("""
<style>
    /* This style block within gr.HTML is not best practice for external CSS.
       It's redundant if the rules are already in custom_css.
       Keeping it for now as it was in your provided code,
       but ideally, these rules would be consolidated into `custom_css`. */
    .top-icons {
        display: flex;
        justify-content: flex-end;
        gap: 30px;
        padding: 10px 20px;
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
    #info_modal {
        display: none;
        position: fixed;
        top: 10%; /* Adjusted to be higher on screen */
        left: 50%;
        transform: translateX(-50%);
        background: #2C2C2C;
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.4);
        z-index: 9999;
        max-width: 650px; /* Increased max-width further for more space */
        width: 90%; /* Maintain percentage width for responsiveness */
        max-height: 80vh; /* Set a max height relative to viewport height */
        overflow-y: auto; /* Enable internal scrolling if content overflows */
        box-sizing: border-box; /* Include padding in height calculation */
    }

    #info_modal .disclaimer {
        font-size: 0.9em;
        color: #aaaaaa;
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px dashed #444;
        text-align: center;
    }

    #info_modal .disclaimer a {
        color: #4a90e2;
        text-decoration: none;
        font-weight: bold;
    }

    #info_modal .disclaimer a:hover {
        text-decoration: underline;
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
    .prompt-btn {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 16px;
        cursor: pointer;
        font-size: 14px;
        transition: background-color 0.2s ease, transform 0.2s ease;
    }
    .prompt-btn:hover {
        background-color: #357ABD;
        transform: scale(1.05);
    }
</style>

<div class="top-icons">
    <button id="info_icon" title="About Agent">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M11 9h2V7h-2v2zm0 8h2v-6h-2v6zm1-16C5.48 1 1 5.48 1 11s4.48 10 10 10 10-4.48 10-10S16.52 1 12 1zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
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

<script>
    // Show info modal on icon click
    document.getElementById('info_icon').onclick = () => {
        document.getElementById('info_modal').style.display = 'block';
    };
    document.getElementById('close_modal').onclick = () => {
        document.getElementById('info_modal').style.display = 'none';
    };

    // MODIFIED: Script to get Gradio File URL for download and update the link
    // Use a short delay to ensure Gradio has rendered the hidden file component
    document.addEventListener('DOMContentLoaded', (event) => {
        setTimeout(() => {
            const fileContainer = document.getElementById('pdf_file_component'); // The hidden gr.File component's container
            if (fileContainer) {
                // Find the actual <a> tag that Gradio creates for download within the hidden component
                const gradioDownloadLink = fileContainer.querySelector('.file-preview a');
                const customDownloadIcon = document.getElementById('download_icon');

                if (gradioDownloadLink && customDownloadIcon) {
                    customDownloadIcon.href = gradioDownloadLink.href;
                    // The 'download' attribute is already set on the custom icon's HTML
                    // customDownloadIcon.download = "Akshay_Abraham_Resume.pdf"; // This line is not strictly needed if already in HTML
                }
            }
        }, 500); // 500ms delay: Give Gradio time to fully render its components.
    });
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
                    "cite", "citation", "citations"
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