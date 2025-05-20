import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your CV and split into chunks
with open("cv.txt", "r") as f:
    cv_chunks = [line.strip() for line in f if line.strip()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
cv_embeddings = model.encode(cv_chunks)

def answer_question(question):
    # Embed the question
    question_embedding = model.encode([question])
    
    # Find most relevant CV chunk
    similarities = cosine_similarity(question_embedding, cv_embeddings)
    best_index = np.argmax(similarities)
    
    return cv_chunks[best_index]

iface = gr.ChatInterface(
    answer_question,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask about my background", container=False, scale=7),
    title="My CV Assistant",
    description="Ask me anything about my professional experience and qualifications",
    theme="soft",
    examples=["What's your educational background?", "What programming languages do you know?"],
    cache_examples=True,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

iface.launch()