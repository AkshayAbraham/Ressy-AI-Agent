# Install dependencies (Hugging Face Spaces will handle this automatically)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load Phi-2 model & tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define your professional profile
profile = {
    "name": "Arun Sharma",
    "title": "Machine Learning Engineer",
    "skills": ["Python", "PyTorch", "TensorFlow", "NLP", "Computer Vision", "LLMs", "Docker", "REST APIs"],
    "experience": "3 years in AI/ML",
    "education": "B.Tech in CS",
    "strengths": ["Fast learner", "Team player", "Good communicator"]
}

# Function to generate chatbot responses
def chat_with_agent(job_description):
    prompt = f"""
    You are Arun Sharma, a skilled Machine Learning Engineer. A recruiter has shared a job description:
    "{job_description}"
    
    Respond naturally, smartly, and persuasively. Highlight relevant skills, even if they only partially match. Explain why you're a strong candidate.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create Gradio chat interface
interface = gr.Interface(
    fn=chat_with_agent,
    inputs=gr.Textbox(label="Paste Job Description"),
    outputs=gr.Textbox(label="AI Agent Response"),
    title="AI-Powered Professional Agent",
    description="Let recruiters chat with your AI agent instead of reading a resume!",
)

# Launch the chatbot
interface.launch()