# app.py
import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from time import time
import re

# Your professional profile
PROFILE = """
Name: Arun Sharma
Title: Machine Learning Engineer
Skills: Python, PyTorch, TensorFlow, NLP, Computer Vision, LLMs, Docker, REST APIs
Experience: 3 years in AI/ML
Education: B.Tech in CS
Strengths: Fast learner, team player, good communicator
"""

# FLAN-T5 optimized prompt template
SYSTEM_PROMPT = """Respond as Arun's professional agent. Rules:
1. Always start with "As Arun's agent"
2. Use third-person (Arun has...)
3. Be concise (2 sentences max)
4. End with a question

Arun's Profile:
{profile}

Conversation History:
{history}

Recruiter: {input}
Agent:"""

# Initialize model
print("⚙️ Loading FLAN-T5...")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
print("✅ Model ready!")

def format_history(history):
    return "\n".join([f"Recruiter: {msg}\nAgent: {response}" for msg, response in history])

def generate_response(message, history):
    try:
        start_time = time()
        
        prompt = SYSTEM_PROMPT.format(
            profile=PROFILE,
            history=format_history(history),
            input=message
        )
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing
        if not response.startswith("As Arun's agent"):
            response = f"As Arun's agent, {response}"
            
        response = (
            response.replace("I have", "Arun has")
            .replace("I am", "Arun is")
            .replace("my", "Arun's")
            .replace("I ", "Arun ")
            .replace(" me", " him")
        )
        
        print(f"⏱️ Response in {time()-start_time:.1f}s")
        return "", history + [(message, response)]
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return "", history + [(message, "Please ask again")]

# Gradio interface with working Enter key
with gr.Blocks(js="""() => {
    const txtArea = document.querySelector('textarea');
    txtArea.placeholder = "Type message (Enter to send, Shift+Enter for new line)";
    txtArea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.querySelector('button.primary').click();
        }
    });
}""") as demo:
    gr.Markdown("""# 🤖 Arun's Professional Agent (FLAN-T5)""")
    
    chatbot = gr.Chatbot(height=350)
    msg = gr.Textbox(label="Message", lines=2)
    submit_btn = gr.Button("Send", variant="primary")
    clear_btn = gr.Button("Clear")
    
    msg.submit(generate_response, [msg, chatbot], [msg, chatbot])
    submit_btn.click(generate_response, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

demo.launch()