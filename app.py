import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

# --- Gradio UI Block ---
with gr.Blocks() as demo:
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

    chatbot = gr.Chatbot(type="messages", height=400)
    with gr.Row(equal_height=True):
        with gr.Column(scale=10):
            msg = gr.Textbox(label="Ask a question about Akshay's profile", container=False)
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