import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Static profile (simulate CV)
profile = """
Akshay Abraham is a Computing graduate with experience in software development, cheminformatics, and bioinformatics.
He is proficient in Python, Django, SQL, and machine learning. He has worked on AI lab assistants, bioinformatics pipelines,
and job tracking Chrome extensions. He has a Masters in Computing from Sheffield Hallam University and a BSc in Chemistry from Madras Christian College.
"""

def generate_response(user_input, chat_history):
    prompt = f"You are an AI assistant helping recruiters learn about this candidate:\n\n{profile}\n\nUser: {user_input}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask about your experience, skills..."),
    outputs="text",
    title="Job Assistant (Flan-T5)",
    description="Ask about the candidate's background. Runs on Hugging Face CPU using Flan-T5."
)

iface.launch()
