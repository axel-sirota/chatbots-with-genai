import os
import numpy as np
import requests
import faiss
import json
from pathlib import Path
import torch
import gradio as gr
from transformers import AutoModel, AutoTokenizer, logging
from huggingface_hub import InferenceClient

#-----------------Constants---------------------
CHUNKED_DOCUMENTS_PATH = Path("./chunked_documents.json")
CHUNKED_DOCUMENTS_URL = "https://www.dropbox.com/scl/fi/07wd0zwvz2xcq80hy5f91/chunked_documents.json?rlkey=jwvfpczo4zeyke9j74cdphovi&st=oeqmcfi8&dl=1"
INDEX_PATH = "./faiss_index.idx"
FAISS_INDEX_URL = "https://www.dropbox.com/scl/fi/05ez2886nz5fkkcqsv6hs/faiss_index.idx?rlkey=yil6ollju5smk04upluenqot4&st=yu0oji49&dl=1"
dimension = 384  # Embedding size from MiniLM model
CHAT_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

#------------------Load chunks-------------------
if CHUNKED_DOCUMENTS_PATH.exists():
    print("Loading existing chunked_documents.json...")
    with open(CHUNKED_DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        chunked_documents = json.load(f)
else:
    print("chunked_documents.json does not exist. Trying to download from remote URL...")
    response = requests.get(CHUNKED_DOCUMENTS_URL, allow_redirects=True)
    response.raise_for_status()
    with open(CHUNKED_DOCUMENTS_PATH, "wb") as f:
        f.write(response.content)
    print("Successfully downloaded chunked_documents.json from remote URL.")
    with open(CHUNKED_DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        chunked_documents = json.load(f)
print(f"Total document chunks available: {len(chunked_documents)}")

#------------------FAISS------------------------
if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index from disk...")
    index = faiss.read_index(INDEX_PATH)
    print(f"Total embeddings indexed: {index.ntotal}")
else:
    print("FAISS index does not exist. Trying to download from remote URL...")
    response = requests.get(FAISS_INDEX_URL, allow_redirects=True)
    response.raise_for_status()
    with open(INDEX_PATH, "wb") as f:
        f.write(response.content)
    print("Successfully downloaded FAISS index from remote URL.")
    index = faiss.read_index(INDEX_PATH)
    print(f"Total embeddings indexed: {index.ntotal}")

#-----------------Chatbot & Inference Client-----------------------
logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For retrieval, we load our local MiniLM model.
retrieval_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
retrieval_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
retrieval_model.cpu().eval()

# Initialize the InferenceClient for Mistral.
# Ensure your HF_API_TOKEN is set in your environment.
HF_TOKEN = os.environ.get("HF_API_TOKEN")
client = InferenceClient(token=HF_TOKEN, model=CHAT_MODEL_ID)

# tokenizer for chat template
chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_ID)
#-----------------Functions----------------------

# Generate embeddings for a new query (using the local retrieval model).
def get_query_embedding(query):
    with torch.no_grad():
        inputs = retrieval_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = retrieval_model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().cpu().numpy()
    return embedding

# Retrieve relevant documents based on the query.
def retrieve_documents(query, top_k=4):
    query_embedding = get_query_embedding(query).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = [chunked_documents[idx] for idx in indices[0]]
    return results

# Generate a response using the InferenceClient's streaming chat_completion.
def generate_response_stream(history, max_new_tokens=100, temperature=0.7, top_p=0.95):
    # Call the inference client with streaming disabled.
    resp_stream = client.chat.completions.create(
        model=CHAT_MODEL_ID,
        messages=history,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True
    )
    # resp_stream is iterable; each chunk has delta content
    for chunk in resp_stream:
        # chunk.choices is a list; delta may have new content
        delta = chunk.choices[0].delta
        # delta may have 'content'
        if hasattr(delta, "content") and delta.content is not None:
            yield delta.content

# ---------------- Response Function ----------------
def respond(message: str,
            history: list,
            system_message: str,
            max_tokens: int,
            temperature: float,
            top_p: float):
    """
    This function builds a conversation history (a list of dicts)
    and calls the Mistral inference client with that history.
    It first ensures that the history starts with the system message,
    retrieves relevant context for the new user message,
    and then streams the assistant's reply.
    """
    if not history:
        history = [{"role": "system", "content": system_message}]
    # Retrieve context based on the user query.
    similar_documents = retrieve_documents(message)
    retrieved_text = " ".join(similar_documents)
    input_text = f"User query: {message}\n\nContext:\n{retrieved_text}"
    # Append the new user message to the conversation history.
    history.append({"role": "user", "content": input_text})
    
    # Now stream
    accumulated = ""
    for delta in generate_response_stream(history, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p):
        accumulated += delta
        yield accumulated

    # Append final assistant message
    history.append({"role": "assistant", "content": accumulated})

# ---------------- Gradio Chat Interface ----------------
demo = gr.ChatInterface(
    fn=respond,
    type="messages",  # Conversation history is a list of message dictionaries.
    title="RAG Chatbot via Inference Client",
    description=(
        "A chatbot powered via the Hugging Face Inference API. "
        "It uses local document retrieval to provide context for the conversation. "
        "Adjust parameters below to test and troubleshoot responses."
    ),
    additional_inputs=[
        gr.Textbox(value="You are a helpful AI assistant.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=350, step=10, label="Max new tokens"),
        gr.Slider(minimum=0.0, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
    ]
)

if __name__ == "__main__":
    demo.launch(debug=True, share=True, server_name="0.0.0.0", server_port=8080)
