import faiss
import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify, render_template

# Step 1: Load your dataset from 'dataset.json'
with open('dataset.json', 'r') as f:
    data = json.load(f)  # Load the JSON content

# Extract the "text" field from each document
documents = [item["text"] for item in data]

# Step 2: Load a local embedding model using HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Step 3: Generate embeddings for your dataset
document_embeddings = embedding_model.embed_documents(documents)
document_embeddings = np.array(document_embeddings)

# Step 4: Create FAISS index and add embeddings
index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Initialize FAISS
index.add(document_embeddings)  # Add embeddings to the FAISS index

# Step 5: Initialize LangChain components
docs = [Document(page_content=doc) for doc in documents]
docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})
index_to_docstore_id = {i: i for i in range(len(docs))}

# Initialize FAISS-based vector store
vector_store = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding_model
)

# Define function to call Ollama's REST API with streaming
def call_ollama_llama(prompt):
    """Call the Ollama REST API to generate a response using the LLaMA 3.1 model."""
    try:
        url = "http://localhost:11434/api/generate"  # Update with correct Ollama API URL if necessary
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3.1",
            "prompt": prompt
        }

        response = requests.post(url, json=data, headers=headers, stream=True)

        if response.status_code == 200:
            # Accumulate the streamed response
            collected_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = line.decode('utf-8')
                    try:
                        chunk_json = json.loads(chunk)
                        collected_response += chunk_json.get("response", "")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {chunk}")

            return collected_response
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return "Error in generating response from Ollama."

    except Exception as e:
        print(f"Exception occurred: {e}")
        return "Exception occurred while calling Ollama."

# Flask web interface
app = Flask(__name__)
history = []  # Global conversation history

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot Interface</title>
    </head>
    <body>
        <h1>Chat with AI</h1>
        <form id="chatForm">
            <label for="userInput">You:</label>
            <input type="text" id="userInput" name="message" required>
            <button type="submit">Send</button>
        </form>
        <div id="chatHistory">
            <h2>Chat History</h2>
            <pre id="historyContent"></pre>
        </div>

        <script>
            document.getElementById('chatForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const userInput = document.getElementById('userInput').value;
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userInput })
                });
                const data = await response.json();
                const chatHistory = document.getElementById('historyContent');
                chatHistory.textContent += `You: ${userInput}\nAI: ${data.response}\n`;
                document.getElementById('userInput').value = '';  // Clear the input field
            });
        </script>
    </body>
    </html>
    '''

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']

    # Append user input to the conversation history
    history.append(f"You: {user_input}")

    # Prepare the context from history
    context = "\n".join(history)

    # Generate a prompt including the conversation history
    prompt = f"Conversation so far:\n{context}\n\nAI:"

    # Call Ollama API to get the response
    response = call_ollama_llama(prompt)

    # Append the AI's response to the history
    history.append(f"AI: {response}")

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
