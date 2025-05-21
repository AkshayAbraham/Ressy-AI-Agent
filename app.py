# app.py
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
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

# Optimized system prompt for TinyLlama
SYSTEM_PROMPT = """[INST] <<SYS>>
You are Arun Sharma's professional agent. Rules:
1. Always say "As Arun's agent" first
2. Use third-person (Arun has...)
3. Be concise (2-3 sentences max)
4. End with a question

Arun's Profile:
{profile}
<</SYS>>

{history}
Recruiter: {input} [/INST]
Agent:"""

def load_model():
    print("🚀 Loading TinyLlama...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float32
    )

print("⚙️ Initializing...")
pipe = load_model()
print("✅ Ready!")

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
        
        output = pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=pipe.tokenizer.eos_token_id
        )
        
        response = output[0]['generated_text'].split("Agent:")[-1].strip()
        response = re.sub(r"Recruiter:.*", "", response, flags=re.DOTALL).strip()
        
        # Ensure proper disclosure and third-person
        if not response.startswith("As Arun's agent"):
            response = f"As Arun's agent, {response}"
            
        response = (
            response.replace("I have", "Arun has")
            .replace("I am", "Arun is")
            .replace("my", "Arun's")
            .replace("I ", "Arun ")
        )
        
        print(f"⏱️ Generated in {time()-start_time:.1f}s")
        return "", history + [(message, response)]
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return "", history + [(message, "Please ask again"))]

# Gradio interface with Enter key fix
with gr.Blocks(title="Arun's Agent (TinyLlama)") as demo:
    gr.Markdown("""# 🤖 Arun Sharma's Professional Agent""")
    
    chatbot = gr.Chatbot(height=350)
    msg = gr.Textbox(
        label="Message",
        placeholder="Type message + Enter to send",
        lines=2
    )
    clear = gr.Button("Clear Chat")
    
    # Configure submit actions
    msg.submit(
        fn=generate_response,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# Required for Hugging Face Spaces
demo.launch()