import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer # NEW
import faiss # NEW
import numpy as np # NEW
import os
import json

# Create cache directory if it doesn't exist
os.makedirs('.gradio/cached_examples', exist_ok=True)

# --- Model Choice: FLAN-T5-base ---
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- Embedding Model for Semantic Search (NEW) ---
# Choose a small, efficient embedding model suitable for CPU
embedding_model_name = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)

# --- Your Professional Profile Data ---
akshay_profile_data = {
    "name": "Akshay Abraham",
    "summary": "Akshay Abraham is a highly accomplished AI/ML and Full-stack developer with a strong background in Cheminformatics. He is known for delivering quantifiable results and innovative solutions across diverse technical domains.",
    "core_strengths": {
        "AI/ML": "Built a Dialogflow AI Assistant saving researchers 15+ hours/month by automating 50+ weekly chemistry queries.",
        "Cheminformatics": "Developed a drug discovery pipeline improving efficiency by 30% and boosted druglikeness prediction accuracy by 30% with Ligand Explorer.",
        "Full-stack Development": "Created production-ready tools using Django, Docker, and Azure; developed a Job Tracker Chrome Extension used by 1000+ students."
    },
    "key_projects": {
        "Ligand Explorer": "A project that boosted druglikeness prediction accuracy by 30% through advanced computational methods.",
        "AI Lab Assistant": "A Dialogflow AI Assistant that automated 50+ weekly chemistry queries, saving researchers 15+ hours/month.",
        "Job Tracker Chrome Extension": "A popular Chrome Extension developed that is used by over 1000+ students for job application tracking, streamlining their job search process."
    },
    "technical_skills": [
        "AI/ML", "Cheminformatics", "Full-stack Development", "Django", "Docker", "Azure",
        "Natural Language Processing", "Machine Learning", "Deep Learning", "Python",
        "Database Management (SQL, NoSQL)", "Cloud Computing", "API Development",
        "Data Analysis", "Algorithm Design", "Software Architecture"
    ],
    "quantifiable_achievements": [
        "Saved researchers 15+ hours/month with Dialogflow AI Assistant.",
        "Improved drug discovery pipeline efficiency by 30%.",
        "Boosted druglikeness prediction accuracy by 30% with Ligand Explorer.",
        "Automated 50+ weekly chemistry queries.",
        "Developed Job Tracker Chrome Extension used by 1000+ students."
    ],
    "work_experience": [
        {
            "company_name": "Innovative AI Labs",
            "title": "Senior AI/ML Engineer",
            "duration": "Jan 2023 - Present",
            "responsibilities": "Leads development of scalable AI solutions, including enterprise Dialogflow integrations."
        },
        {
            "company_name": "ChemTech Pharma",
            "title": "Cheminformatics Scientist",
            "duration": "Jul 2020 - Dec 2022",
            "responsibilities": "Designed and implemented drug discovery pipelines, focusing on predictive modeling."
        },
        {
            "company_name": "DevSolutions Co.",
            "title": "Full-stack Developer",
            "duration": "Jun 2018 - Jun 2020",
            "responsibilities": "Built and maintained web applications, including a popular Chrome extension for job tracking."
        }
    ],
    "value_proposition": "Akshay Abraham is a unique blend of scientific rigor and cutting-edge technical execution, consistently delivering solutions that not only meet but exceed expectations, driving efficiency and innovation."
}

# --- Prepare Documents for Semantic Search (NEW) ---
# Create a list of all meaningful text chunks from your profile
document_chunks = []
# Add summary and value proposition
document_chunks.append(profile_data["summary"])
document_chunks.append(profile_data["value_proposition"])

# Add core strengths
for strength_area, description in profile_data["core_strengths"].items():
    document_chunks.append(f"{strength_area}: {description}")

# Add key projects
for project_name, description in profile_data["key_projects"].items():
    document_chunks.append(f"Project {project_name}: {description}")

# Add technical skills (as a single chunk or individual skills)
document_chunks.append(f"Technical Skills: {', '.join(profile_data['technical_skills'])}")

# Add quantifiable achievements
for achievement in profile_data["quantifiable_achievements"]:
    document_chunks.append(f"Achievement: {achievement}")

# Add work experience
for job in profile_data["work_experience"]:
    document_chunks.append(f"Work Experience: {job['company_name']} as {job['title']} ({job['duration']}). Responsibilities: {job['responsibilities']}")

# Create embeddings for all document chunks
print("Creating embeddings for document chunks...")
document_embeddings = embedding_model.encode(document_chunks, convert_to_numpy=True)

# Build a FAISS index for efficient similarity search
index = faiss.IndexFlatL2(document_embeddings.shape[1]) # L2 distance for similarity
index.add(document_embeddings)
print("FAISS index built.")

# --- Retrieval Function using Semantic Search (NEW) ---
def retrieve_info_semantic(query, top_k=3):
    # Encode the query
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    
    # Reshape for FAISS search
    query_embedding = np.array([query_embedding]) 
    
    # Perform similarity search
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the top_k most relevant document chunks
    relevant_chunks = [document_chunks[idx] for idx in indices[0]]
    
    # Combine and ensure context fits within token limits
    final_context = "\n".join(relevant_chunks)
    if len(tokenizer.encode(final_context)) > 300: 
        final_context = tokenizer.decode(tokenizer.encode(final_context)[:300], skip_special_tokens=True)
        
    return final_context

# --- Prompt Engineering (Same as before) ---
def build_prompt(user_input, retrieved_context):
    context_section = f"Here is relevant information about Akshay Abraham's profile:\n{retrieved_context}\n\n" if retrieved_context else ""

    return f"""You are Akshay Abraham's dedicated SENIOR TECHNICAL RECRUITER. Your primary goal is to **aggressively and persuasively pitch Akshay** as an absolutely exceptional, must-hire candidate to potential employers. You are a salesperson, and Akshay is your top product. Highlight his unique value and accomplishments to compel the employer.

{context_section}

Follow these strict rules for your responses:
1.  **ALWAYS start with an IMMEDIATE, HIGH-IMPACT VALUE PROPOSITION** that grabs attention and states why Akshay is indispensable.
2.  **Integrate 2-3 specific, QUANTIFIABLE ACHIEVEMENTS** directly relevant to the user's query and the provided context. These are your proof points – emphasize the tangible impact (e.g., "saved 15+ hours/month," "improved efficiency by 30%").
3.  **Elaborate significantly on his expertise and problem-solving abilities**, connecting them directly to the benefits for the hiring company. Explain *how* his skills translate into value.
4.  **Provide a multi-sentence, detailed response**, typically 4-8 sentences long. Avoid short, single-line answers. Be thorough and compelling.
5.  **Maintain a professional, confident, and enthusiastic tone** throughout.
6.  **ONLY use information provided in the profile data or derivable from it.** Do not invent new facts.

Examples:
User: Why hire Akshay for AI roles?
Assistant: You absolutely need Akshay for your AI team! He's a proven innovator who doesn't just build AI; he delivers production-ready solutions that drive measurable impact, like his Dialogflow assistant that **saved researchers 15+ hours/month** by automating complex queries. His unique blend of deep chemistry knowledge and machine learning expertise means he can not only develop cutting-edge AI tools but also understand and solve intricate domain-specific challenges, making him a strategic asset for real innovation and efficiency gains in your organization.

User: Tell me about his technical skills.
Assistant: Akshay possesses a robust full-stack and MLOps skillset that goes far beyond basic coding – he architects and implements solutions that solve critical business problems from end-to-end. For instance, he developed a full-stack Django pipeline that **improved workflow efficiency by 30%**, demonstrating his ability to create scalable, reliable, and highly impactful systems. His proficiency with Docker and Azure ensures seamless deployment, continuous integration, and robust maintenance of complex applications, guaranteeing your projects are not just built, but deployed effectively and reliably.

User: {user_input}
Assistant:"""

def generate_response(user_input):
    try:
        # Step 1: Retrieve relevant information using semantic search
        retrieved_context = retrieve_info_semantic(user_input) # Changed function call
        
        # Step 2: Build the prompt with the retrieved context
        prompt = build_prompt(user_input, retrieved_context)
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

        # --- Generation Parameters: Optimized for FLAN-T5's behavior ---
        output = model.generate(
            **inputs,
            max_new_tokens=250,
            num_beams=5,
            do_sample=False,
            repetition_penalty=1.3,
            early_stopping=True,
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Post-processing for FLAN-T5: remove the prompt from the output
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        elif "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

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
    title="Akshay Abraham - Technical Profile Assistant (Semantic RAG)",
    description="Ask any question about Akshay's technical background and achievements. I'm here to highlight why he's the perfect fit!",
    allow_flagging="never"
)

iface.launch(share=False)