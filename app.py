import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Create cache directory if it doesn't exist
os.makedirs('.gradio/cached_examples', exist_ok=True)

# Load FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Optimized profile with quantifiable achievements
profile = """
Candidate Profile: Akshay Abraham
- Core Strengths:
  • AI/ML: Built Dialogflow AI Assistant saving researchers 15+ hours/month
  • Cheminformatics: Developed drug discovery pipeline improving efficiency by 30%
  • Full-stack: Created production-ready tools (Django, Docker, Azure)
- Key Projects:
  • Ligand Explorer: Boosted druglikeness prediction accuracy by 30%
  • AI Lab Assistant: Automated 50+ weekly chemistry queries
  • Job Tracker Chrome Extension: Used by 1000+ students
"""

def build_prompt(user_input):
    return f"""You are a SENIOR TECHNICAL RECRUITER pitching Akshay Abraham. Follow these rules:
1. Always start with a STRONG VALUE PROPOSITION
2. Include 1-2 QUANTIFIABLE ACHIEVEMENTS
3. Keep responses under 3 sentences (concise but powerful)

Examples:
User: Why hire Akshay for AI roles?
Assistant: Akshay delivers production-ready AI solutions - like his Dialogflow assistant that saved researchers 15+ hours/month. His unique blend of chemistry and ML expertise drives real innovation.

User: Tell me about his technical skills.
Assistant: Beyond just coding, Akshay solves business problems. His Django pipeline improved workflow efficiency by 30%, while his Docker deployments ensure reliable scaling.

{profile}

User: {user_input}
Assistant:"""

def generate_response(user_input):
    try:
        prompt = build_prompt(user_input)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            repetition_penalty=1.3,
            early_stopping=True
        )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean up response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response if response else "Akshay brings unique value through his combination of technical skills and domain expertise."
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Let me highlight Akshay's strengths: [Your fallback message here]"

# Create Gradio interface without example caching
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask about skills, projects, or achievements..."),
    outputs="text",
    title="Akshay Abraham - Technical Profile Assistant",
    description="Ask any question about Akshay's technical background and achievements",
    allow_flagging="never"
)

iface.launch(share=False, ssr_mode=False)  # Disable SSR to prevent cache issues