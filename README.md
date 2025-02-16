RAG-RAPTOR

RAG-RAPTOR is a Retrieval-Augmented Generation (RAG) system designed to enhance responses with relevant document retrieval.

Installation Guide

Follow the steps below to set up and run the project.

1. Download and install Ollama from https://ollama.com/.

2. Pull Model with Ollama
```sh
ollama pull model_name
```

3. Install Dependencies and Navigate to Project Directory
```sh
conda create -n chatbot python==3.10
conda activate chatbot
cd RAG-RAPTOR
pip install -r requirements.txt
```

4. Run the Chatbot

Ensure Ollama is running, then start the Streamlit application:
```sh
streamlit run chatbot.py
```
License

Contact

For any issues or questions, feel free to open an issue or contribute to the repository.