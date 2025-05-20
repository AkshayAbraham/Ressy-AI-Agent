import gradio as gr
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your custom data
documents = SimpleDirectoryReader("data").load_data()

# Set up a small model like Phi-2
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Build service context with the model
llm = HuggingFaceLLM(context_window=2048, max_new_tokens=256, model=model, tokenizer=tokenizer)
service_context = ServiceContext.from_defaults(llm=llm)

# Create index
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Define chat function
def chat_fn(message, history):
    response = chat_engine.chat(message)
    return str(response)

# Launch chatbot UI
gr.ChatInterface(
    chat_fn,
    title="Ask Me About My Work",
    description="Chat with my AI agent to learn about my skills, experience, and projects.",
    theme="default"
).launch()
