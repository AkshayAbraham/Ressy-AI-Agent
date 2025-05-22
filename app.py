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

# Setup
os.makedirs('.gradio/cached_examples', exist_ok=True)
load_dotenv()

# Initialize components
embedding_model = setup_embedding_model("sentence-transformers/all-mpnet-base-v2")
my_resume = load_text_data("data/resume.txt")
chunks = [chunk.strip() for chunk in my_resume.split("---") if chunk.strip()]
db = Chroma.from_texts(chunks, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# CSS to hide ALL progress indicators
custom_css = """
/* Hide all progress indicators and response time displays */
.progress-bar, .progress, .eta, .status, .toast {
    display: none !important;
}

/* Main styling */
body, .gradio-container {
    background-color: #1A1A1A;
    color: white;
}

#chatbot {
    background-color: #1A1A1A;
    border: none;
    padding: 0;
}

/* Input container */
#input_container {
    position: relative;
    background-color: #2C2C2C;
    border: 1px solid #444;
    border-radius: 25px;
    margin: 10px 0 20px;
}

/* Input textbox */
#input_textbox {
    width: 100%;
    background: transparent;
    color: white;
    padding: 12px 50px 12px 15px;
    border: none;
}

/* Send button */
#send_button {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background: #4a90e2;
    color: white;
    border: none;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}

/* Loading spinner */
#send_button.loading {
    background: transparent;
    border: 3px solid rgba(255,255,255,0.2);
    border-top: 3px solid #4a90e2;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: translateY(-50%) rotate(0deg); }
    100% { transform: translateY(-50%) rotate(360deg); }
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")
    
    chatbot = gr.Chatbot(height=400)
    loading_state = gr.State(False)

    with gr.Column(elem_id="input_container"):
        msg = gr.Textbox(
            placeholder="Ask me anything about Akshay's profile...",
            elem_id="input_textbox",
            container=False
        )
        submit = gr.Button("➤", elem_id="send_button")

    def respond(message, chat_history, loading_state):
        # Clear input and show loading
        yield "", chat_history, True
        
        # Process message
        relevant_excerpts = semantic_search(message, retriever)
        bot_message = resume_chat_completion(
            client, "llama-3.3-70b-versatile", message, relevant_excerpts
        )
        
        # Update chat and hide loading
        chat_history.append((message, bot_message))
        yield "", chat_history, False

    # Event handlers
    submit.click(
        respond,
        [msg, chatbot, loading_state],
        [msg, chatbot, loading_state],
        queue=True
    )
    
    msg.submit(
        respond,
        [msg, chatbot, loading_state],
        [msg, chatbot, loading_state],
        queue=True
    )
    
    # Loading state toggle
    loading_state.change(
        lambda x: gr.Button(visible=not x),
        loading_state,
        submit,
        js="""
        function(start) {
            const btn = document.getElementById('send_button');
            if (start) {
                btn.classList.add('loading');
                btn.innerHTML = '';
            } else {
                btn.classList.remove('loading');
                btn.innerHTML = '➤';
            }
            return start;
        }
        """
    )

if __name__ == "__main__":
    demo.launch()