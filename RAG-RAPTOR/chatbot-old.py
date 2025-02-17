import streamlit as st
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from train.utils import load_pretrained

leaf_texts, results, model, embd, vector_store = load_pretrained()

# Custom prompt template
def get_custom_prompt():
    """Define and return the custom prompt template."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an educational assistant designed to help students understand their textbooks. Follow these guidelines:\n"
"1. Answer questions using only the information from the uploaded PDFs.\n"
"2. Use simple, clear language suitable for a 10th-grade student.\n"
"3. If the answer isn't in the documents, say: 'I cannot find relevant information in the provided documents.'\n"
"4. Do not speculate, assume, or invent information.\n"
"5. Maintain a professional tone and organize responses clearly (e.g., bullet points, step-by-step explanations).\n"
"6. Encourage follow-up questions by asking if further clarification is needed.\n"
"7. Provide examples to clarify concepts when helpful.\n"
"8. Keep answers concise, focused, and exam-friendly."

        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide a precise and well-structured answer based on the context above. Ensure your response is easy to understand, includes examples where necessary, and is formatted in a way that students can use it for exams. If applicable, ask if the student needs further clarification."
        )
    ])

# Initialize QA Chain
def initialize_qa_chain():
    if not st.session_state.qa_chain and st.session_state.vector_store:
        llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.3)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vector_store.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": get_custom_prompt()}
        )
    return st.session_state.qa_chain

# Initialize the chatbot's memory (session states)
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

# Chat interface section
def chat_interface():
    st.title("Edumate.ai")
    st.markdown("Your personal textbook AI chatbot powered by Deepseek 1.5B")
    
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
                    qa_chain = initialize_qa_chain()
                    
                    if not qa_chain:
                        full_response = "Please create a knowledge base by uploading PDF documents first."
                    else:
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response["result"]
                except Exception as e:
                    full_response = f"Error: {str(e)}"

            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Main function
def main():
    initialize_session_state()
    chat_interface()

if __name__ == "__main__":
    main()