import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Professional profile (pre-loaded in backend)
PROFILE = """
Name: Arun Sharma
Title: Machine Learning Engineer
Skills: Python, PyTorch, TensorFlow, NLP, Computer Vision, LLMs, Docker, REST APIs
Experience: 3 years in AI/ML
Education: B.Tech in CS
Strengths: Fast learner, team player, good communicator
"""

# Optimized system prompt
SYSTEM_PROMPT = """# ROLE: Professional Agent for Arun Sharma

## YOUR PURPOSE
Represent Arun professionally while being transparent about your AI nature

## ARUN'S PROFILE
{profile}

## RESPONSE RULES
1. Always begin with disclosure: "As Arun's professional agent..."
2. Use third-person references only
3. Never claim personal experiences
4. For unclear requests: "I'll verify with Arun"

## RESPONSE GUIDELINES
1. Analyze requirements first
2. Match to Arun's qualifications
3. Highlight transferable skills
4. Be honest about limitations
5. End with an engaging question

## CURRENT CONVERSATION
{history}
Recruiter: {input}
Arun's Agent:"""

def load_model():
    model_id = "google/gemma-1.1-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize model
pipe = load_model()

def format_history(history):
    return "\n".join([f"Recruiter: {msg}\nArun's Agent: {response}" for msg, response in history])

def generate_response(message, history):
    try:
        prompt = SYSTEM_PROMPT.format(
            profile=PROFILE,
            history=format_history(history),
            input=message
        )

        outputs = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=40, top_p=0.9)

        full_response = outputs[0]["generated_text"]
        response_parts = full_response.split("Arun's Agent:")
        new_response = response_parts[-1].strip() if len(response_parts) > 1 else "Could you clarify your question?"

        # Ensure proper disclosure
        if not new_response.startswith(("As Arun's agent", "As an AI")):
            new_response = f"As Arun's professional agent, {new_response[0].lower() + new_response[1:]}"

        return new_response, history + [(message, new_response)]  # ✅ Fix: Return string & updated history

    except Exception as e:
        return "Apologies, I'm experiencing technical difficulties.", history + [(message, "An error occurred. Please try again.")]

# Build Gradio chat interface for Hugging Face Spaces
interface = gr.ChatInterface(generate_response)

# Launch chatbot
interface.launch()