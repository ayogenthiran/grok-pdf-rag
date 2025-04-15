import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load API keys from .env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    <context>

    Question: {input}
    """
)

# App Title
st.title("📄 RAG Document Q&A with Groq + LLaMA3")

# PDF Check
if not os.path.exists("research_papers") or not os.listdir("research_papers"):
    st.warning("⚠️ Please upload PDFs to the 'research_papers' folder before continuing.")

# Create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

# Input prompt
user_prompt = st.text_input("🔍 Enter your query about the research paper:")

# Embedding button
if st.button("🧠 Create Document Embeddings"):
    create_vector_embedding()
    st.success("✅ Vector database is ready!")

# Query section
if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write(f"🕒 Response Time: {round(time.process_time() - start, 2)} seconds")

    # Display answer
    st.markdown("### 💡 Answer:")
    st.write(response['answer'])

    # Show document matches
    with st.expander("📚 Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.markdown("---")

elif user_prompt:
    st.warning("❗ Please embed documents first before asking a question.")
