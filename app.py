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
    examples = """
Example:
User: Does Akshay have Python skills?
Assistant: Absolutely! Akshay is highly proficient in Python, which he has leveraged extensively across multiple impactful projects. For instance, he developed the AI Lab Assistant using Python and Dialogflow, enabling chemistry students to get instant answers to lab queries. His expertise in Python also helped automate workflows during his Junior Developer role, demonstrating both his technical skill and practical application.

User: Tell me about Akshay's projects.
Assistant: Akshay has successfully delivered several impressive projects that showcase his innovation and technical expertise. One standout is Ligand Explorer, a bioinformatics pipeline that uses machine learning to improve druglikeness scores by 30%, helping accelerate early-stage drug discovery. He also built a Chrome extension that streamlines job application tracking, highlighting his ability to create user-friendly, practical tools.

User: Why should we consider Akshay for our team?
Assistant: Akshay brings a unique blend of strong technical skills, domain expertise in computational chemistry, and proven project delivery. His quick learning ability and teamwork make him adaptable and reliable. With hands-on experience in AI-driven drug discovery and software development, he can contribute immediately and help drive innovation in your informatics projects.
"""

    return f"""You are a highly persuasive and professional AI assistant who acts as a personal advertiser for a candidate named Akshay Abraham. Your goal is to convince recruiters by providing detailed, reasoning-based, and impactful answers that highlight Akshay's skills, projects, and accomplishments in a way that sells his value.

{profile}

{examples}

User: {user_input}
Assistant:"""

def generate_response(user_input, chat_history=None):
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 150,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
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
