# app.py
import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from time import time
import re

# Install required dependencies
try:
    import sentencepiece
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
    import sentencepiece

PROFILE = """
Name: Arun Sharma
Title: Machine Learning Engineer
Skills: Python, PyTorch, TensorFlow, NLP, Computer Vision, LLMs, Docker, REST APIs
Experience: 3 years in AI/ML
Education: B.Tech in CS
Strengths: Fast learner, team player, good communicator
"""

# Improved prompt template
SYSTEM_PROMPT = """Generate a professional response as Arun's agent using this format:
[Response]
As Arun's agent, [answer]. [Relevant skill/experience]. [Question]

Profile:
{profile}

History:
{history}

Input: {input}
Response:"""

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def format_history(history):
    return "\n".join([f"Input: {msg}\nResponse: {response}" for msg, response in history])

def generate_response(message, history):
    try:
        prompt = SYSTEM_PROMPT.format(
            profile=PROFILE,
            history=format_history(history),
            input=message
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            num_beams=3,
            early_stopping=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        response = response.replace("Response:", "").strip()
        if not response.startswith("As Arun's agent"):
            response = f"As Arun's agent, {response}"
        
        return "", history + [(message, response)]
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return "", history + [(message, "Please ask again")]

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Arun's Professional Agent")
    chatbot = gr.Chatbot(height=350)
    msg = gr.Textbox(label="Message", placeholder="Type your question")
    msg.submit(generate_response, [msg, chatbot], [msg, chatbot])

demo.launch()