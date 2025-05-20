import gradio as gr
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI # Keep this import
from llama_index.llms.huggingface import HuggingFaceLLM # Keep this for local fallback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM # Only for local fallback
import torch

# --- Configuration for LLM and Embeddings ---

# 1. Hugging Face Inference API (Recommended for conversational LLM on free tier)
# This requires HF_TOKEN to be set as a secret in your Hugging Face Space settings.
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    print("HF_TOKEN found. Attempting to use Hugging Face Inference API for LLM...")
    try:
        # Changed model_name to a highly reliable instruction-tuned model
        llm = HuggingFaceInferenceAPI(
            model_name="mistralai/Mistral-7B-Instruct-v0.2", # <-- UPDATED MODEL
            token=HF_TOKEN,
            temperature=0.1, # Lower temperature for more factual, less creative responses
            max_new_tokens=256 # Good practice to limit response length
        )
        print(f"Successfully initialized LLM using Hugging Face Inference API with model: {llm.model_name}")
    except Exception as e:
        print(f"Error initializing Hugging Face Inference API LLM: {e}")
        print("Falling back to a small local model (if available) or dummy LLM.")
        llm = None # Set to None so the fallback logic can kick in
else:
    print("HF_TOKEN not found. Falling back to a small local LLM (may be limited).")
    llm = None

# --- Fallback for LLM (if HF_TOKEN is not set or Inference API fails) ---
if llm is None: # Only attempt local if Inference API failed or HF_TOKEN was missing
    # Attempt to load a *very small causal language model* that *might* fit on CPU Basic.
    # Note: Even these are challenging for the free tier due to memory/CPU.
    # "facebook/opt-125m" is a very small causal model, but its quality is limited.
    local_model_name = "facebook/opt-125m" # This is a very small *causal* model
    try:
        print(f"Attempting to load local LLM: {local_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_name)
        model = AutoModelForCausalLM.from_pretrained(local_model_name,
                                                     torch_dtype=torch.float32,
                                                     low_cpu_mem_usage=True)

        model.to("cpu") # Ensure model is on CPU

        llm = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.1, "do_sample": True},
            tokenizer=tokenizer,
            model=model,
            device_map="cpu",
        )
        print(f"Successfully loaded local LLM: {local_model_name}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load any functional LLM (local or Inference API). Details: {e}")
        print("The chatbot will not be able to answer questions correctly.")
        # Define a basic LLM stub that adheres to the LlamaIndex LLM interface for the Assertion to pass
        from llama_index.core.llms import CustomLLM, CompletionResponse

        class FallbackLLM(CustomLLM):
            def complete(self, prompt, **kwargs):
                return CompletionResponse(text="I'm sorry, my language model could not be loaded. Please check the Space logs and ensure your HF_TOKEN is set for better performance.")
            async def acomplete(self, prompt, **kwargs):
                return self.complete(prompt, **kwargs)
            def chat(self, messages, **kwargs):
                # For chat, simply use the last message content as the prompt
                return self.complete(str(messages[-1].content))


        llm = FallbackLLM()


# 2. Embedding model (Crucial for RAG)
print("Loading embedding model BAAI/bge-small-en-v1.5...")
try:
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("Embedding model loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load embedding model. Details: {e}")
    class DummyEmbedding:
        def get_text_embedding(self, text):
            return [0.0] * 384
        async def aget_text_embedding(self, text):
            return self.get_text_embedding(text)
    embed_model = DummyEmbedding()

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
    print(f"CRITICAL ERROR: During data loading or index creation: {e}")
    class ErrorChatEngine:
        def chat(self, message):
            return "I'm sorry, I encountered a critical error loading my knowledge base. Please check the 'data' directory and logs."
    chat_engine = ErrorChatEngine()


# --- Gradio Interface ---
def chat_with_me(message, history):
    try:
        # Gradio's chat interface passes 'history', but LlamaIndex's chat_engine
        # handles its own internal history, so we just pass the new message.
        response = chat_engine.chat(message)
        return str(response)
    except Exception as e:
        return f"An error occurred during chat: {e}. Please check Space logs for details."


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