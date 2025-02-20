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

# Load pretrained models and data
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

# Initialize session state
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
        st.session_state.show_intro = True  # Show intro page by default

# Session history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Load pretrained components
leaf_texts, results, model, embd, vector_store = load_pretrained()
retriever = vector_store.as_retriever()

# Contextualizing user queries
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

# Constructing the QA chain
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

# Wrap chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

##########web design start############################
# Chat interface
def show_intro():
    # Update global styles
    st.set_page_config(page_title="Shallowseek", page_icon="logo.svg", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.image("logo.svg",width=100)
    st.markdown("""
        <style>
            .stApp {
                background-color: #E3F2FD !important;  /* Lighter blue background */
            }
            .css-1d391kg {
                background-color: #E3F2FD !important;
            }
            .stButton>button {
                background-color: white !important;
                color: black !important;
                font-size: 20px;
                padding: 15px 32px;
                border-radius: 10px;
                border: 2px solid #1976D2;
                margin: 20px auto;
                display: block;
                width: 200px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                background-color: #f8f9fa !important;
                border-color: #1565C0;
            }
            .team-text {
                position: fixed;
                top: 0;
                left: 0;
                font-size: 24px;
                font-weight: bold;
                color: #1976D2;
                z-index: 1000;
                padding: 10px 20px;
                background-color: #E3F2FD;
            }
            .question-container {
                width: 100vw;
                margin-left: -50vw;
                left: 50%;
                position: relative;
                overflow: hidden;
                margin-top: 10px;
                margin-bottom: 10px;
            }
            .scrolling-questions {
                display: inline-flex;
                flex-wrap: nowrap;
                animation: scroll 20s linear infinite;  /* Set scroll speed */
                animation-fill-mode: none;
                white-space: nowrap;
            }
            .question-box {
                background-color: #f5f5f5;
                border-radius: 8px;
                padding: 15px 25px;
                margin: 0 20px;
                min-width: 250px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: inline-block;
                color: #333;
                cursor: pointer;
                margin-bottom: 10px;
            }
            @keyframes scroll {
                0% {
                    transform: translateX(0); /* Start from the left side */
                }
                50% {
                    transform: translateX(-50vw); /* Scroll halfway through */
                }
                100% {
                    transform: translateX(-100vw); /* End at the far left */
                }
            }
            .scrolling-questions:hover {
                animation-play-state: paused; /* Pause the scrolling when hovered */
            }
            h1, h3 {
                color: #1976D2 !important;
            }
            p {
                color: #333 !important;
            }
            .stMarkdown {
                color: #333;
            }
        </style>
    """, unsafe_allow_html=True)

    # Add "Team" text to the top left corner
    st.markdown('<div class="team-text">Team</div>', unsafe_allow_html=True)

    # Display title with no line breaks
    st.markdown("<h1 style='text-align: center; white-space: nowrap;'>Climate-Mind: Your Climate Consultant</h1>", unsafe_allow_html=True)

    st.markdown("### About the Model")
    st.markdown("""Climate-mind is a free, specialized climate consultant powered by a robust climate science and policy database.
    With strong analytical reasoning, it provides data-driven insights and solutions to tackle complex climate changes.
    """)

    # Center the "Start Now" button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Now", use_container_width=True):
            st.session_state.show_intro = False  # End the introduction and go to the chat interface
            st.rerun()  # Refresh the page to show the chat interface

    st.markdown("### Let's explore together:")

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

    # Create scrolling questions in one continuous loop
    st.markdown(f"""
        <div class="question-container">
            <div class="scrolling-questions">
                {"".join([f'<div class="question-box">{q}</div>' for q in questions * 3])}
            </div>
        </div>
    """, unsafe_allow_html=True)



# Chat interface for messages
def chat_interface():
    if st.session_state.show_intro:
        show_intro()
        return

    # Display chat messages and inputs here
    
    st.set_page_config(page_title="Shallowseek", page_icon="logo.svg", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.markdown("""
        <style>
            .stApp {
                background-color: #E3F2FD !important;  /* Lighter blue background */
            }
            h1, h3 {
                color: #1976D2 !important;
            }
            .stChatMessageAvatar {
                width: 40px !important;  /* Adjust width */
                height: 40px !important; /* Adjust height */
                border-radius: 50%;      /* Keep circular shape */
                object-fit: cover;
            }
        </style>
    """, unsafe_allow_html=True)
    st.image("logo.svg",width=120)
    
    st.markdown("<h1 style='text-align: center; white-space: nowrap;'>Climate-Mind: Your Climate Consultant</h1>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about climate policies"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant",avatar="logo.svg"):
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
#######################web design end#####################################
# Main function
def main():
    initialize_session_state()
    if st.session_state.show_intro:
        show_intro()
    else:
        chat_interface()

if __name__ == "__main__":
    main()