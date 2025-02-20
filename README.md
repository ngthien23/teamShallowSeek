RAG-RAPTOR

RAG-RAPTOR is a Retrieval-Augmented Generation (RAG) system designed to enhance responses with relevant document retrieval.

Installation Guide

Follow the steps below to set up and run the project.

1. Download and install Ollama from https://ollama.com/.

2. Pull Model with Ollama (to use another model, modify model name in climate-mind.py)
```sh
ollama pull nomic-embed-text
ollama pull deepseek-r1:1.5b
```

3. Install Dependencies and Navigate to Project Directory
```sh
conda create -n chatbot python==3.10
conda activate chatbot
cd chatbot
pip install -r requirements.txt
```

4. Run the Chatbot
To use our vector embeddings, extract chromadb.rar in folder chatbot.

Ensure Ollama is running in another terminal:
```sh
ollama serve
```

Start the Streamlit application:
```sh
streamlit run climate-mind.py
```
License

Contact

For any issues or questions, feel free to open an issue or contribute to the repository.