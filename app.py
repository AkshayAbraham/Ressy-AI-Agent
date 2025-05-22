import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5 base model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Shortened static profile (preloaded only once)
profile = """
Candidate Profile: Akshay Abraham
- Computing Graduate with strong background in cheminformatics, bioinformatics, and computational chemistry.
- Skilled in Python, C#, Dart, SQL, Django, ML (TensorFlow, Keras, Dialogflow), RDKit, and DFT tools like Gaussian.
- Experience: Software Developer, Research Assistant (AI-driven drug discovery), and chemistry informatics projects.
- Projects: AI Lab Assistant (Dialogflow), Ligand Explorer (Bioinformatics), Android App (Object Detection), Chrome Extension (Job Tracking).
- Tools: VS Code, PyCharm, Azure, Docker, MySQL, MongoDB, GitHub.
- Soft Skills: Quick learner, Analytical thinker, Strong communicator, Team-oriented.
"""

# Function to build optimized prompt
def build_prompt(user_input):
    return f"""You are a helpful assistant answering questions about a candidate's profile.

{profile}

User: {user_input}
Assistant:"""

# Response generator
def generate_response(user_input, chat_history=None):
    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Strip leading profile/echo if model repeats part of prompt
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
