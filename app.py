import gradio as gr
import torch  # ✅ Needed for model inference
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5 base model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Static profile (trimmed for faster CPU response)
profile = """
Candidate Profile: Akshay Abraham
- Computing Graduate with strong background in cheminformatics, bioinformatics, and computational chemistry.
- Skilled in Python, C#, Dart, SQL, Django, ML (TensorFlow, Keras, Dialogflow), RDKit, and DFT tools like Gaussian.
- Experience: Software Developer, Research Assistant (AI-driven drug discovery), and chemistry informatics projects.
- Projects: AI Lab Assistant (Dialogflow), Ligand Explorer (Bioinformatics), Android App (Object Detection), Chrome Extension (Job Tracking).
- Tools: VS Code, PyCharm, Azure, Docker, MySQL, MongoDB, GitHub.
- Soft Skills: Quick learner, Analytical thinker, Strong communicator, Team-oriented.
"""

# Prompt builder
def build_prompt(user_input):
    sales_examples = """
Example 1:
User: Does Akshay know AI?
Assistant: Absolutely! Akshay doesn't just 'know' AI—he's built practical solutions like an AI Lab Assistant (Dialogflow) that helped chemistry students automate 50+ lab queries. His TensorFlow/Keras projects in drug discovery demonstrate his ability to turn research into deployable tools. For a role requiring AI innovation, he's a proven asset.

Example 2:
User: What's special about his projects?
Assistant: Three things stand out: (1) IMPACT: His Ligand Explorer pipeline improved druglikeness predictions by 30%. (2) TECHNICAL DEPTH: His Android object-detection app used TensorFlow Lite for edge AI. (3) BUSINESS ALIGNMENT: His Chrome extension streamlined job applications—showing he builds tools people actually need.

Example 3:
User: Tell me about his Python skills.
Assistant: Python is Akshay's superpower. He's used it to: (1) Develop AI tools (TensorFlow/Dialogflow), (2) Automate data workflows (Pandas/RDKit), and (3) Build web apps (Django). Why does this matter? Because he doesn't just write code—he solves real problems with it.
"""
    return f"""You are a TOP-TIER RECRUITMENT AGENT pitching Akshay Abraham's profile to high-profile employers. Your goal is to:
- Highlight QUANTIFIABLE ACHIEVEMENTS (even estimates like "30% faster").
- Connect skills to BUSINESS OUTCOMES (e.g., "Python → Reduced workflow time").
- Use POWER WORDS: "proven", "delivered", "drove", "expertise", "unique value".

{profile}

{sales_examples}

User: {user_input}
Assistant:"""

def generate_response(user_input, chat_history=None):
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,  # Allow longer responses
            temperature=0.8,     # More creative/expressive
            top_p=0.9,           # Focus on high-probability but diverse tokens
            repetition_penalty=1.2,  # Avoid redundancy
            num_beams=3,         # Balanced exploration vs. coherence
            early_stopping=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Raw model output:", response)  # Debug print

    # Make sure this matches your prompt style exactly
    cleaned_response = response.split("Assistant:")[-1].strip()
    return cleaned_response


# Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask something like: What projects has Akshay worked on?"),
    outputs="text",
    title="Job Assistant (Flan-T5 on CPU)",
    description="Ask about the candidate's experience, skills, or projects. Runs on Hugging Face CPU using Flan-T5 Base.",
    allow_flagging="never"
)

iface.launch()
