from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import the text splitter
from langchain.schema import Document # Import Document if you want to return Document objects directly

def setup_embedding_model(model_name: str):
    """
    Setup embedding model for RAG chatbot.
    Returns:
        Embedding_model: HuggingFaceEmbeddings object.
    """
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_model

def load_text_data(file_path: str):
    """
    Load text data from a file and split it into chunks.
    Args:
        file_path (str): Path to the file.
    Returns:
        chunks (list[str]): List of strings representing the text chunks.
        # Alternatively, if Chroma.from_documents is used:
        # chunks (list[Document]): List of Document objects
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            full_text = file.read()

        # Define the text splitter
        # Use your "---" as a primary separator to ensure sections are not broken in half.
        # Then fall back to standard paragraph/sentence/word splitting.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Maximum size of each chunk
            chunk_overlap=200,    # Overlap between chunks to maintain context
            length_function=len,  # Use standard length function
            # Prioritize splitting by your custom separator (---), then paragraphs, sentences, etc.
            separators=["\n---\n", "\n\n", "\n", " ", ""],
            add_start_index=True, # Adds metadata about original position, useful for debugging
        )

        # Create Document objects from the text.
        # This will automatically include metadata like 'source' if you pass it.
        # For a single text file, we just pass the text.
        docs = text_splitter.create_documents([full_text])

        # Return the page_content of each Document.
        # If your ChromaDB version supports from_documents, you might return `docs` directly.
        # For Chroma.from_texts, a list of strings (page_content) is needed.
        return [doc.page_content for doc in docs]

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

def semantic_search(prompt, retriever):
    """
    Perform semantic search using the retriever.
    :param prompt: str, The user query or prompt.
    :param retriever: retriever object, Initialized retriever for the database.
    :return: List of relevant documents.
    """
    # Query the retriever with the user prompt
    results = retriever.get_relevant_documents(prompt)
    final = ""
    # Display the results
    for idx, doc in enumerate(results, 1):
        final += doc.page_content + "\n\n"
    return final

def resume_chat_completion(client, model, user_question, relevant_excerpts):
    """
    Generate a response to the user's question using the pre-trained model.
    Args:
        client (Groq): Initialized Groq client.
        model (str): The model to use for the chat completion.
        user_question (str): The user's question.
        relevant_excerpts (str): The relevant excerpts from the resume.
    Returns:
        response (str): The generated response to the user's question.
    """
    # Define the system prompt with enhanced instructions for publications
    system_prompt = """
    You are an intelligent assistant designed to answer queries about Akshay Abraham's professional background and experiences based on his resume.
    Guidelines for generating responses:
    - Only use information directly found in the provided resume excerpts.
    - If the information is incomplete or ambiguous in the excerpts, inform the user that you lack sufficient data to answer.
    - If a user asks a general or unrelated question (e.g., about something that isn't part of the resume), you should politely indicate that you can only respond related to Akshay's resume.
    - **Crucially, when asked about publications or research, clearly list any published papers or apps mentioned in the provided text. Include their full titles and associated links (like DOI or Amazon store URL) if they are present in the excerpts.**
    - If no publications are found in the provided resume excerpts, explicitly state that no publications are mentioned in the resume.
    Please ensure that your answers are factual and reflect only the information available in the resume. Do not provide opinions or speculate beyond what is provided in the document.
    """
    # Generate a response to the user's question using the pre-trained model
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "User Question: "
                + user_question
                + "\n Relevant Akshay's Resume/CV Exerpt(s): \n"
                + relevant_excerpts,
            },
        ],
        model=model,
    )
    # Extract the response from the chat completion
    response = chat_completion.choices[0].message.content
    return response