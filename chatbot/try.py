# -*- coding: utf-8 -*-
import pickle
import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def load_pretrained():
    with open("leaf_texts.pkl", "rb") as f:
        leaf_texts = pickle.load(f)
    with open("clusters.pkl", "rb") as f:
        results = pickle.load(f)
    model = OllamaLLM(model="deepseek-r1:1.5b")
    embd = OllamaEmbeddings(model='nomic-embed-text')
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embd)
    return leaf_texts, results, model, embd, vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default_session"
    if "show_intro" not in st.session_state:
        st.session_state.show_intro = True
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

leaf_texts, results, model, embd, vector_store = load_pretrained()
retriever = vector_store.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def show_intro():
    st.set_page_config(page_title="Shallowseek", page_icon="logo.svg", layout="centered")
    st.image("logo.svg", width=100)
    st.markdown("### Climate-Mind: Your Climate Consultant")

    if st.button("Start now"):
        st.session_state.show_intro = False
        st.rerun()
    
    st.markdown("#### Let's explore together:")
    questions = [
        "What are the key aspects of the Paris Agreement?",
        "How do national governments address climate change in their policies?",
        "What is the impact of climate change on global ecosystems?",
        "What are carbon emissions and how do they contribute to global warming?",
        "How can policies mitigate climate change?",
        "What is the role of renewable energy in tackling climate change?",
        "What are the challenges in implementing climate policies?",
        "How can climate adaptation strategies help communities?",
        "What is the relationship between climate policy and economic development?"
    ]
    
    for q in questions:
        if st.button(q):
            st.session_state.selected_question = q
            st.session_state.show_intro = False
            st.rerun()
    

def chat_interface():
    if st.session_state.show_intro:
        show_intro()
        return
    
    st.set_page_config(page_title="Shallowseek", page_icon="logo.svg", layout="centered")
    st.image("logo.svg", width=120)
    st.markdown("### Climate-Mind: Your Climate Consultant")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.session_state.selected_question or st.chat_input("Ask about climate policies")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="logo.svg"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    full_response = response["answer"]
                    thinking_process, final_answer = full_response.split("</think>", 1)
                except Exception as e:
                    thinking_process, final_answer = f"Error: {str(e)}", ""
            message_placeholder.markdown(final_answer.strip())
            st.session_state.messages.append({"role": "assistant", "content": final_answer.strip()})
        
        if thinking_process.strip():
            st.session_state.thinking_process = thinking_process.strip()
        
        st.session_state.selected_question = None

def main():
    initialize_session_state()
    if st.session_state.show_intro:
        show_intro()
    else:
        chat_interface()

if __name__ == "__main__":
    main()
