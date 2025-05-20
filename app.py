from llama_index import VectorStoreIndex, SimpleDirectoryReader
import gradio as gr

# Load data from the "data/" directory
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

def chat_with_me(message, history):
    response = chat_engine.chat(message)
    return str(response)

chatbot_ui = gr.ChatInterface(
    fn=chat_with_me,
    title="💬 Ask Me Anything About My Work!",
    description="This is my personal AI agent. Ask me about my skills, experience, or projects.",
    theme="default",
    examples=[
        "Do I have experience with React?",
        "What Python projects have I worked on?",
        "Tell me about my work experience.",
        "Do I know software engineering best practices?"
    ],
)

if __name__ == "__main__":
    chatbot_ui.launch()
