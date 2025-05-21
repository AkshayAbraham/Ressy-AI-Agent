import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load CV data
with open("cv.txt", "r") as f:
    cv_chunks = [line.strip() for line in f if line.strip()]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')
cv_embeddings = model.encode(cv_chunks)

def respond(message, chat_history):
    # Embed the question
    question_embedding = model.encode([message])
    
    # Find most relevant CV chunk
    similarities = cosine_similarity(question_embedding, cv_embeddings)
    best_index = np.argmax(similarities)
    
    response = cv_chunks[best_index]
    chat_history.append((message, response))
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("# My CV Assistant")
    gr.Markdown("Ask me anything about my professional background")
    
    chatbot = gr.Chatbot(height=300)
    msg = gr.Textbox(label="Your question")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()