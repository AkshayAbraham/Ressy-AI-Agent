import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Static profile (simulate CV)
profile = """
Name: Akshay Abraham
Title: Computing Graduate | Informatics Developer | ML Enthusiast

Summary:
Computing graduate with a strong background in computational chemistry, cheminformatics, and bioinformatics. Experienced in software development, molecular modeling, and data management. Passionate about solving complex problems with technology. Quick learner and collaborative team player with a focus on informatics development and analysis.

Education:
- Masters in Computing, Sheffield Hallam University, England (2023–2024) | Merit
  - Project: Machine learning solutions for lab safety (custom vs. no-code platforms)
- B.Sc. in Chemistry, Madras Christian College, India (2018–2021) | First Class
  - Projects: Inexpensive polarimeter design, aspirin replacement with detox drink
  - Roles: Techno Club Convener, NSS Volunteer

Experience:
- Junior Software Developer, MadEmpty, India (Mar 2022 – Nov 2022)
  - Developed and maintained applications using Python, C#, Dart, SQL, Django.
- Research Assistant, Computational Chemistry Lab, India (May 2021 – Nov 2021)
  - Worked on DFT simulations and AI-driven drug discovery using Gaussian software.
- Crew Member, McDonald's, UK (Jan 2023 – Sep 2024)
  - Provided high-quality customer service in a fast-paced environment.

Certifications:
- Advanced SQL (Kaggle, Apr 2025)
- LIMS and Information Systems (CDC, Apr 2025)
- Database Foundations (Oracle, Mar 2025)
- Cheminformatics (Oct 2018)

Projects:
- AI Lab Assistant: Python + Dialogflow bot for chemistry lab queries.
- Ligand Explorer: ML-based bioinformatics pipeline improving druglikeness scores by 30%.
- Job Application Manager: Chrome extension for job tracking via Google Sheets integration.
- Event Web App: PHP + real-time chat, auth, bulk email, and role management.
- Django Tutor Platform: Azure-hosted student-tutor matching system.
- Catalyst App: Android app for lab apparatus detection using TensorFlow.

Skills:
- Programming: Python, C#, Dart, SQL, HTML, CSS, JavaScript, Django, Flutter
- ML/AI: TensorFlow, Keras, YOLO, Dialogflow, OpenCV, RDKit
- Web & App Dev: Android Studio, Azure, Docker, Kubernetes, MySQL, MongoDB, NoSQL
- Tools: GitHub, VS Code, PyCharm, Adobe Illustrator, Figma, Google Colab
- Computational Chemistry: Gaussian, Schrödinger, AutoDock, PyMOL, DFT, Drug Design

Soft Skills:
Project Management, Teamwork, Communication, Logical Thinking, Time Management, Adaptability, Fast Learner

Publications:
- AI-based SARS-CoV-2 Inhibitors (RSC: https://doi.org/10.1039/D4ME00062E)
- Periodic Table Mobile App (Amazon: https://www.amazon.com/dp/B08BYHLBX8)

Languages:
English, Malayalam, Tamil, Hindi
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
