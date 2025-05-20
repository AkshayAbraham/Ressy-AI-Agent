import gradio as gr
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# We will ONLY use HuggingFaceLLM for a local model for now
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- Configuration for Local LLM and Embeddings (For Free Tier) ---

# We are explicitly choosing a very small local LLM that *should* fit on CPU Basic.
# Its quality will be limited, but it avoids Inference API issues.
local_llm_model_name = "facebook/opt-125m" # A very small Causal Language Model

print(f"Attempting to load local LLM: {local_llm_model_name}...")
try:
    # Load tokenizer and model for the chosen small LLM
    tokenizer = AutoTokenizer.from_pretrained(local_llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        local_llm_model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU to avoid errors
        low_cpu_mem_usage=True     # Optimize for low CPU memory
    )

    # Ensure the model is moved to CPU
    model.to("cpu")

    llm = HuggingFaceLLM(
        context_window=2048, # Adjust based on model's max context length (OPT-125M is 2048)
        max_new_tokens=256,  # Limit generation length
        generate_kwargs={"temperature": 0.1, "do_sample": True, "top_p": 0.9}, # Added top_p for better generation
        tokenizer=tokenizer,
        model=model,
        device_map="cpu",    # Force CPU usage for free tier
    )
    print(f"Successfully loaded local LLM: {local_llm_model_name}")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load local LLM '{local_llm_model_name}'. Details: {e}")
    print("The chatbot will not be able to answer questions correctly.")
    # Define a basic LLM stub that adheres to the LlamaIndex LLM interface
    from llama_index.core.llms import CustomLLM, CompletionResponse

    class FallbackLLM(CustomLLM):
        def complete(self, prompt, **kwargs):
            return CompletionResponse(text="I'm sorry, my language model could not be loaded. Please check the Space logs for errors during LLM loading.")
        async def acomplete(self, prompt, **kwargs):
            return self.complete(prompt, **kwargs)
        def chat(self, messages, **kwargs):
            return self.complete(str(messages[-1].content)) # Simple fallback for chat

    llm = FallbackLLM()


# 2. Embedding model (Crucial for RAG)
# This model converts text into numerical vectors for searching your data.
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
# Settings.chunk_size = 512 # You might want to adjust chunk size for embedding


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