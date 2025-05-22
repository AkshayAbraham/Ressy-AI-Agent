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
/* Overall background color for the body and the main Gradio container */
body, .gradio-container {
    background-color: #1A1A1A !important; /* A very dark gray, almost black */
}

/* Rounded corners and background for the chatbot display area */
#chatbot {
    border-radius: 15px !important;
    background-color: #1A1A1A !important; /* Set to the same as body background */
    overflow: hidden;
    border: 1px solid #333333;
}

/* Styling for the individual message bubbles within the chatbot */
.gr-message-bubble {
    border-radius: 12px !important;
    padding: 10px 15px !important;
}

.gr-message-bubble.gr-message-user {
    background-color: #007bff !important; /* A standard blue for user messages */
    color: white !important;
}

.gr-message-bubble.gr-message-bot {
    background-color: #444444 !important; /* Darker gray for bot messages */
    color: white !important;
}

/* Input textbox styling:  Combined input and send button */
#input_row {
    display: flex; /* Use flexbox to position input and button */
    align-items: center; /* Vertically center the elements */
    border-radius: 15px; /* Rounded corners for the entire row */
    background-color: #2c2c2c; /* Background color for the input area */
    border: 1px solid #555555;
    padding-right: 5px; /* Add some padding on the right for the button */
    margin-top: 5px; /* Add a bit of space between the chatbot and input */
}

#input_textbox {
    flex-grow: 1; /* Make the textbox take up remaining space */
    border: none !important; /* Remove the textbox's individual border */
    background-color: transparent !important; /* Make the textbox background transparent */
    box-shadow: none !important; /* Remove any box shadow */
    border-radius: 15px 0 0 15px !important; /* Rounded corners on the left */
}

#input_textbox textarea {
    background-color: transparent !important; /* Ensure textarea is also transparent */
    color: #FFFFFF !important; /* White text for input */
    padding: 10px; /* Add padding inside the textarea */
}

/* Style for the send button (now inside the input row) */
#send_button {
    background-color: #4a90e2 !important; /* Blue send button */
    color: white !important;
    border: none !important;
    border-radius: 0 15px 15px 0 !important; /* Rounded corners on the right */
    padding: 10px 15px !important;
    cursor: pointer; /* Change cursor to pointer on hover */
    height: auto; /* Allow the button to adjust its height */
}

#send_button:hover {
    background-color: #357ABD !important; /* Darker blue on hover */
}


/* Hide the Clear button */
.clear-button {
    display: none !important;
}

/* Placeholder styling (mimicking Gemini) */
#input_textbox textarea::placeholder {
  color: #999; /* Light gray placeholder text */
  font-style: italic; /* Italic placeholder text */
}


"""

# --- Gradio UI Block ---
# Ensure this 'with' block is at the top level of indentation (no leading spaces)
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
    chatbot = gr.Chatbot(type="messages", height=400, elem_id="chatbot")

    # This gr.Row contains the textbox and the send button
    with gr.Row(elem_id="input_row", equal_height=True):
        with gr.Column(scale=10):
            # The textbox now has a placeholder and no default label
            msg = gr.Textbox(
                label="",
                placeholder="Ask me anything about Akshay's profile...",
                container=False,
                elem_id="input_textbox"
            )
        with gr.Column(scale=1):
            # The submit button is now inside this row and has a custom ID
            submit = gr.Button(value="➤", size="sm", elem_id="send_button")
    # The clear button is intentionally removed from here
    # clear = gr.ClearButton([msg, chatbot], size="sm")

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
        bot_message = resume_chat_completion(
            client, "llama-3.3-70b-versatile", message, relevant_excerpts
        )

        # Append to history and return both history and empty string for textbox
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    # Bind submit button and textbox to the respond function
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot]) # Allow submission with Enter key

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()