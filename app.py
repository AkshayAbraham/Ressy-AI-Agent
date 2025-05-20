import gradio as gr
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --- Configuration for Free/Local LLM and Embeddings ---
# IMPORTANT: For free tier, we avoid paid APIs like OpenAI.
# We'll use a small, open-source model from Hugging Face.

# Option A: Use Hugging Face Inference API (recommended for slightly better models, requires HF_TOKEN)
# This leverages Hugging Face's hosted inference endpoints, which are generally faster
# and can run larger models than what you can fit on a free CPU Space directly.
# You'll need a Hugging Face Access Token with read permissions (it's free to generate one).
# Add it as a secret named HF_TOKEN in your Hugging Face Space settings.

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    print("Using Hugging Face Inference API for LLM...")
    llm = HuggingFaceInferenceAPI(
        model_name="HuggingFaceH4/zephyr-7b-alpha",
        token=HF_TOKEN,
        temperature=0.1
    )
else:
    print("HF_TOKEN not found, falling back to a very small local LLM...")
    model_name = "distilbert-base-uncased-distilled-squad"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        llm = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.1, "do_sample": True},
            tokenizer=tokenizer,
            model=model,
            device_map="cpu",
        )
    except Exception as e:
        print(f"Error loading local LLM {model_name}: {e}")
        print("Falling back to a dummy LLM. The chatbot will not function correctly.")
        class DummyLLM:
            def complete(self, prompt):
                return "Error: LLM could not be loaded. Please check your configuration and available resources."
            def chat(self, messages):
                return "Error: LLM could not be loaded. Please check your configuration and available resources."
        llm = DummyLLM()


# 2. Embedding model (Crucial for RAG)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- Set global LlamaIndex settings ---
Settings.llm = llm
Settings.embed_model = embed_model

# --- Data Loading and Indexing ---
print("Loading documents from 'data' directory...")
try:
    documents = SimpleDirectoryReader("data").load_data()
    print(f"Loaded {len(documents)} documents.")
    print("Creating VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(documents)
    print("Index created.")
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    print("Chat engine initialized.")
except Exception as e:
    print(f"Error during data loading or index creation: {e}")
    class ErrorChatEngine:
        def chat(self, message):
            return "I'm sorry, I encountered an error loading my knowledge base. Please check the 'data' directory and logs."
    chat_engine = ErrorChatEngine()

# --- Gradio Interface ---
def chat_with_me(message, history):
    try:
        response = chat_engine.chat(message)
        return str(response)
    except Exception as e:
        return f"An error occurred during chat: {e}"

chatbot_ui = gr.ChatInterface(
    fn=chat_with_me,
    title="💬 Ask Me Anything About My Work!",
    description="This is my personal AI agent. Ask me about my skills, experience, or projects.",
    theme="default",
    examples=[
        "Do I have experience with React?",
        "What Python projects have I worked on?",
        "Tell me about my work experience.",
        "Do I know software engineering best practices?"
    ],
)

if __name__ == "__main__":
    chatbot_ui.launch()