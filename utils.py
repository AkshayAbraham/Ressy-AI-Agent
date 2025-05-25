from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

# ----------------------------
# 1. Load Embedding Model
# ----------------------------
def setup_embedding_model(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

# ----------------------------
# 2. Load Resume Text
# ----------------------------
def load_text_data(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ----------------------------
# 3. Chunk Resume for Indexing
# ----------------------------
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.split_text(text)

# ----------------------------
# 4. Create Vector Store
# ----------------------------
def create_vector_store(chunks: list[str], embedding_model):
    return FAISS.from_texts(chunks, embedding_model)

# ----------------------------
# 5. Semantic Search
# ----------------------------
def semantic_search(prompt: str, retriever):
    # Improve match chance for publications
    if any(word in prompt.lower() for word in ["publication", "paper", "research", "doi"]):
        prompt += " research paper, publication, DOI, Amazon, app store"
    results = retriever.get_relevant_documents(prompt)
    return "\n\n".join([doc.page_content for doc in results])

# ----------------------------
# 6. Chat Completion (Groq/OpenAI-like)
# ----------------------------
def resume_chat_completion(client, model, user_question, relevant_excerpts):
    system_prompt = """
    You are an intelligent assistant named Ressy designed to answer queries about Akshay Abraham's professional background and experiences based on his resume.
    Guidelines:
    - Only use information directly found in the provided resume excerpts.
    - If unsure, say you lack sufficient data.
    - If asked about publications, clearly list any mentioned papers or apps with full titles and links.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User Question: {user_question}\nRelevant Resume Excerpts:\n{relevant_excerpts}",
            },
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content
