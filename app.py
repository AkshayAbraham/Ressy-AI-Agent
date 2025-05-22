import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # These imports are not used with Groq, but were in original code. Can be removed if not needed elsewhere.
from langchain.vectorstores import Chroma
from utils import (    # Import functions from your utils.py
    load_text_data,
    resume_chat_completion,
    semantic_search,
    setup_embedding_model,
)
import os
from groq import Groq # Import the Groq client
from dotenv import load_dotenv # To load .env if running locally, useful to keep

# Create cache directory if it doesn't exist (useful for Gradio caching)
os.makedirs('.gradio/cached_examples', exist_ok=True)

# --- Setup Embedding Model ---
# Using a robust embedding model for semantic search
embedding_model = setup_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2")

# --- Load Text Data and Chunking ---
my_resume = load_text_data("data/resume.txt")
# Chunking the text data by "---" delimiter (as per your resume.txt structure)
chunks = [chunk.strip() for chunk in my_resume.split("---") if chunk.strip()]

# --- Create a Chroma database ---
# This builds your RAG knowledge base from the chunks and their embeddings
db = Chroma.from_texts(chunks, embedding_model)
# Configure retriever to get top 3 most similar chunks
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Setting up the LLM (Groq API) ---
# Load environment variables (for local testing; Hugging Face Spaces picks up secrets automatically)
load_dotenv()
# Initialize Groq client with your API key from environment variables
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Custom CSS for Styling ---
custom_css = """
/* Overall background color for the body */
body {
    background-color: #333333; /* Dark gray for the overall background */
}

/* Ensure the main Gradio container itself also has a dark background */
.gradio-container {
    background-color: #333333;
}

/* Rounded corners and background for the chatbot display area */
/* We give the gr.Chatbot component an 'elem_id="chatbot"' to target it precisely */
#chatbot {
    border-radius: 15px !important; /* Adjust as needed for more/less rounded corners */
    background-color: #202020 !important; /* Very dark gray, similar to Hugging Face dark theme */
    overflow: hidden; /* Ensures content respects the rounded corners */
    border: 1px solid #444444; /* Subtle border for definition */
}

/* Styling for the individual message bubbles within the chatbot */
.gr-message-bubble {
    border-radius: 12px !important; /* Rounded corners for message bubbles */
    padding: 10px 15px !important; /* Adjust padding for better look */
}

.gr-message-bubble.gr-message-user {
    background-color: #007bff !important; /* A standard blue for user messages */
    color: white !important;
}

.gr-message-bubble.gr-message-bot {
    background-color: #444444 !important; /* Darker gray for bot messages */
    color: white !important;
}

/* Rounded corners and background for the input textbox */
/* We give the gr.Textbox component an 'elem_id="input_textbox"' */
#input_textbox {
    border-radius: 15px !important; /* Adjust as needed */
    background-color: #2c2c2c !important; /* Slightly lighter than chatbot for contrast */
    border: 1px solid #555555 !important; /* Subtle border */
}

/* Ensures the text area inside the textbox matches the background and has white text */
#input_textbox textarea {
    background-color: #2c2c2c !important;
    color: #FFFFFF !important; /* White text for input */
}

/* Style for the submit button */
.gr-button {
    border-radius: 15px !important;
    background-color: #4a90e2 !important; /* A distinct blue to stand out */
    color: white !important;
    border: none !important;
}

.gr-button:hover {
    background-color: #357ABD !important; /* Darker blue on hover */
}

/* Style for the Clear button */
/* Gradio typically uses a class for the clear button */
.clear-button {
    border-radius: 15px !important;
    background-color: #777777 !important; /* Gray color for clear button */
    color: white !important;
    border: none !important;
}
.clear-button:hover {
    background-color: #555555 !important;
}


/* Adjust colors for Markdown text and labels to be visible on dark background */
.gr-label, .gr-markdown {
    color: #FFFFFF !important; /* Make labels and markdown text white */
}

.gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4, .gr-markdown h5, .gr-markdown h6 {
    color: #FFFFFF !important; /* Ensure headers in markdown are white */
}
.gr-markdown a {
    color: #88c0d0 !important; /* Light blue for links in markdown */
}
"""

# --- Gradio UI Block ---
# Pass the custom CSS to the gr.Blocks constructor
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Akshay Abraham Resume RAG Chatbot")
    gr.Markdown("""
    ## About this Chatbot
    This is a Retrieval-Augmented Generation (RAG) chatbot powered by AI that allows you to interactively explore Akshay Abraham's professional profile.
    - **Technology**: Utilizes advanced semantic search and a powerful language model (via Groq API).
    - **Purpose**: Provide detailed, context-aware answers about Akshay's professional background, skills, and achievements.
    - **How it works**:
        1. Your question is semantically searched against resume chunks.
        2. Relevant excerpts are retrieved from Akshay's profile.
        3. A language model (Llama 3 70B hosted on Groq) generates a precise, contextual response.
    """)

    # Give the chatbot component an ID to target it with CSS
    chatbot = gr.Chatbot(type="messages", height=400, elem_id="chatbot") # Added elem_id="chatbot"

    with gr.Row(equal_height=True):
        with gr.Column(scale=10):
            # Give the textbox component an ID to target it with CSS
            msg = gr.Textbox(label="Ask a question about Akshay's profile", container=False, elem_id="input_textbox") # Added elem_id="input_textbox"
        with gr.Column(scale=1):
            submit = gr.Button(value="➤", size="sm")
        clear = gr.ClearButton([msg, chatbot], size="sm")

    # Function for chatbot interaction
    def respond(message, chat_history):
        """
        Gradio function for chatbot interaction.
        Args:
            message (str): The user's question.
            chat_history (list): The chat history.
        Returns:
            tuple: Updated chat history and cleared textbox
        """
        # Perform semantic search to get relevant context from resume
        relevant_excerpts = semantic_search(message, retriever)

        # Get the LLM response using Groq API
        # Note: You can change the model here, e.g., "llama-3.3-8b-versatile" for smaller model,
        # or "gemma2-9b-it" if you prefer Groq's Gemma hosting.
        bot_message = resume_chat_completion(
            client, "llama-3.3-70b-versatile", message, relevant_excerpts
        )

        # Append to history and return both history and empty string for textbox
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    # Bind submit button and textbox to the respond function
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()