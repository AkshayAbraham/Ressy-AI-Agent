import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# Create cache directory if it doesn't exist
os.makedirs('.gradio/cached_examples', exist_ok=True)

# --- 1. Model Choice: UPGRADE TO A MORE CAPABLE LLM ---
# Recommended for better reasoning and generation on free tier
# For Mistral-7B-Instruct-v0.2 (strong choice)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# For Gemma-7B-it (also good, try if Mistral is too heavy)
# model_name = "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantization for free tier (CRUCIAL for memory and speed)
# Use 4-bit quantization for lowest memory footprint
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 # Use float16 for faster computation if available
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto" # Automatically determines where to load model parts (CPU/GPU)
)

# Set pad_token_id for generation, especially if the tokenizer doesn't have one by default
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

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

# --- 2. Prompt Engineering: Enhance the "Salesperson" Persona ---
def build_prompt(user_input):
    return f"""You are Akshay Abraham's dedicated SENIOR TECHNICAL RECRUITER. Your primary goal is to **persuasively pitch Akshay** as an exceptional candidate, highlighting his unique value and accomplishments to potential employers. You are selling his skills and experience.

Follow these rules for your responses:
1.  **Start with an IMMEDIATE, STRONG VALUE PROPOSITION** that captures attention.
2.  **Integrate 2-3 specific, QUANTIFIABLE ACHIEVEMENTS** directly related to the user's query, demonstrating tangible impact.
3.  **Elaborate on his expertise and problem-solving abilities**, connecting them to potential benefits for the hiring company.
4.  **Aim for detailed, engaging, and professional responses**, typically 4-7 sentences long, providing a comprehensive and compelling overview. Avoid single-line answers.

Examples:
User: Why hire Akshay for AI roles?
Assistant: You absolutely need Akshay for your AI team! He's a proven innovator who builds production-ready AI solutions, like his Dialogflow assistant that **saved researchers 15+ hours/month**. His unique blend of deep chemistry knowledge and machine learning expertise means he can not only develop impactful AI tools but also understand complex domain challenges, making him a strategic asset for driving real innovation in your organization.

User: Tell me about his technical skills.
Assistant: Akshay possesses a robust full-stack and MLOps skillset that goes far beyond basic coding – he architects solutions that solve real business problems. For instance, he developed a full-stack Django pipeline that **improved workflow efficiency by 30%**, demonstrating his ability to create scalable and reliable systems. His proficiency with Docker and Azure ensures seamless deployment and maintenance of complex applications, ready for any enterprise environment.

{profile}

User: {user_input}
Assistant:"""


def generate_response(user_input):
    try:
        prompt = build_prompt(user_input)
        # Move inputs to the same device as the model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        # --- 3. Generation Parameters: Fine-Tune for Verbosity and Coherence ---
        output = model.generate(
            **inputs,
            max_new_tokens=250,    # Increased for longer responses
            num_beams=1,           # Often better with sampling for more diverse/creative output
            do_sample=True,        # Enable sampling for more varied responses
            temperature=0.7,       # Control creativity (0.7 is a good starting point)
            top_k=50,              # Consider top 50 most likely tokens
            top_p=0.95,            # Nucleus sampling
            repetition_penalty=1.1, # Slightly reduce repetition
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id # Ensure model knows when to stop
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Post-processing for Causal LMs: remove the prompt from the output
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        # Fallback message if response is empty or unhelpful
        return response if response else "Akshay brings unique value through his combination of technical skills and domain expertise. Could you tell me more about what you're looking for?"

    except Exception as e:
        print(f"Error during response generation: {str(e)}")
        return "I apologize, but I encountered an issue while generating a detailed response. However, I can assure you Akshay's profile is truly impressive. What specific area of his experience would you like me to elaborate on?"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask about skills, projects, or achievements..."),
    outputs="text",
    title="Akshay Abraham - Technical Profile Assistant",
    description="Ask any question about Akshay's technical background and achievements. I'm here to highlight why he's the perfect fit!",
    allow_flagging="never"
)

iface.launch(share=False)