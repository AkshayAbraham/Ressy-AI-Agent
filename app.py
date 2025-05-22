import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Profile (trimmed for focus)
profile = """
Candidate Profile: Akshay Abraham
- Skills: Python (TensorFlow, Django), C#, Dart, SQL, RDKit, Gaussian.
- Experience: 
  • Software Developer: Built AI Lab Assistant (Dialogflow) saving 15+ hours/month.
  • Research Assistant: Developed drug discovery tools improving efficiency by 30%.
- Projects: Ligand Explorer (30% better druglikeness), Chrome Extension (1000+ users).
- Tools: Docker, Azure, GitHub, VS Code.
"""

# Sales-optimized prompt builder
def build_prompt(user_input):
    return f"""You are a TOP-TIER RECRUITER pitching Akshay Abraham. Your rules:
1. BE PERSUASIVE: Highlight achievements with numbers (e.g., "30% faster").
2. ANSWER THE QUESTION DIRECTLY first, then add 2-3 key points.
3. USE POWER WORDS: "proven", "delivered", "expertise".

Examples:
User: Why hire Akshay for AI roles?
Assistant: Akshay delivers practical AI solutions—like his Dialogflow Lab Assistant that automated 50+ weekly queries (saving 15 hours/month). His TensorFlow projects in drug discovery show he turns research into tools.

User: How good is his Python?
Assistant: Python is Akshay's core strength. He built a Django pipeline improving workflows by 30%, plus production-ready AI tools. His code is scalable and well-documented (see GitHub).

{profile}

User: {user_input}
Assistant:"""

# Generation with error handling
def generate_response(user_input):
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    try:
        # Beam search for reliable outputs
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=3,
            repetition_penalty=1.2,
            early_stopping=True,
            do_sample=False
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Fallback if profile leaks
        if "Candidate Profile:" in response:
            response = response.split("Assistant:")[-1].strip()
            if not response:  # If empty, regenerate
                output = model.generate(**inputs, max_new_tokens=200, num_beams=5)
                response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return response.split("Assistant:")[-1].strip() if "Assistant:" in response else response
    
    except Exception as e:
        return f"Let me reframe that: Akshay's key strength is {user_input.split('?')[0]}. For example..."  # Fallback pitch

# Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask: 'What makes Akshay stand out for AI roles?'"),
    outputs="text",
    title="Akshay's AI Sales Agent",
    description="Ask about skills, projects, or achievements. Powered by FLAN-T5.",
    examples=[
        ["Why should we hire Akshay?"],
        ["Tell me about his Python expertise."],
        ["What's his most impressive project?"]
    ]
)

iface.launch(share=False)  # Set share=True for public link