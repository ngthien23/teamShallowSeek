import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def load_pretrained():
    with open("leaf_texts.pkl", "rb") as f:
        leaf_texts = pickle.load(f)
    with open("clusters.pkl", "rb") as f:
        results = pickle.load(f)
    model = OllamaLLM(model="deepseek-r1:1.5b")
    embd = OllamaEmbeddings(model='nomic-embed-text')
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embd)
    
    return leaf_texts, results, model, embd, vector_store

# Post-processing function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize chatbot session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

# Chat interface section
def chat_interface():
    st.title("Team ShallowSeek")
    st.markdown("Climate Policies AI chatbot powered by Deepseek 1.5B")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Fetching information..."):
                try:
                    response = rag_chain.invoke(prompt)
                    full_response = response
                except Exception as e:
                    full_response = f"Error: {str(e)}"
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
