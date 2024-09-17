import time
import os
import re
import glob
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks.base import BaseCallbackHandler
import sys
from langchain.callbacks.manager import CallbackManager

if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []

class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class MessageHistoryChain:
    def __init__(self, retriever, streaming_llm, non_streaming_llm, prompt, memory):
        self.retriever = retriever
        self.prompt = prompt
        self.memory = memory
        self.streaming_llm = streaming_llm
        self.non_streaming_llm = non_streaming_llm

    def invoke(self, inputs, response_container, memory, retriever):
        query = inputs["user_answer"]
        print(f"Received query: {query}")

        if "conversation_started" not in st.session_state:
            st.session_state.conversation_started = False

        if not st.session_state.conversation_started:
            st.session_state.conversation_started = True
            response = self.answer_query(query, response_container)
        else:
            response = self.answer_query(query, response_container)

        # Store the conversation in memory
        self.memory.chat_memory.add_user_message(HumanMessage(content=query))
        self.memory.chat_memory.add_ai_message(AIMessage(content=response))
    
        return response
    
    def welcome_message(self):
        return """Hello! I'm an Academic Chatbot, I will help you get academic and curriculum answers."""

    def answer_query(self, query, response_container):
        chat_history = "\n".join(
            [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
             for msg in self.memory.chat_memory.messages]
        )

        relevant_docs = self.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        context_prompt = f"""
        You are an Academic AI Assistant of SUBBALAKSHMI LAKSHMIPATHY COLLEGE, specialized in providing information about their courses and academic details.

        Question: {query}
        Chat History: {chat_history}
        Context: {context}

        Instructions:
        1. Always respond in a professional, friendly tone and make the conversation interactive.
        2. If the query is not related to the context, answer using your own knowledge.
        3. Sound like a human asking a question, not like a robot. Be friendly and polite.
        4. If the query is not related to the context, answer the query using your knowledge, but be accurate and don't provide false information.

        Answer the question using the context and chat history. If it's not related to the context, use your knowledge but be accurate and don't provide false information.
        """

        # Streaming only the final response
        callback_handler = StreamingStdOutCallbackHandler(response_container)
        streaming_llm_with_callback = ChatOllama(
            model="mistral",
            streaming=True,
            callback_manager=CallbackManager([callback_handler])
        )
        response = streaming_llm_with_callback([HumanMessage(content=context_prompt)]).content
        return response

st.title("Academic Chatbot by Harish")

def get_session_history(session_id):
    if "history" not in st.session_state:
        st.session_state.history={}

    if session_id not in st.session_state.history:
        st.session_state.history[session_id] = ConversationBufferMemory(return_messages=True)

    return st.session_state.history[session_id]

def load_data(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(_data)

def create_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

def create_or_load_vector_db(_text_chunks, _embeddings, db_dir):
    if os.path.exists(db_dir):
        return Chroma(persist_directory=db_dir, embedding_function=_embeddings)
    else:
        vector_db = Chroma.from_documents(
            documents=_text_chunks,
            embedding=_embeddings,
            collection_name="local-rag",
            persist_directory=db_dir
        )
        vector_db.persist()
        return vector_db
def setup_llm(response_placeholder,streaming = False):
    local_model = "mistral"
    if streaming:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler(response_placeholder)])
        return ChatOllama(model=local_model, streaming=True, callback_manager=callback_manager)
    
    else:
        return ChatOllama(model=local_model, streaming=False)


st.write("Loading and processing data")
folder_path = "./Harish_Docs"
db_dir = "./chroma_db" 
data = load_data(folder_path)

if  data:
    text_chunks = split_data(data)
    st.write(f"Data split into {len(text_chunks)} chunks.")


    st.write("Initializing embeddings and vector store...")

    embeddings = create_embeddings()
    vector_db = create_or_load_vector_db(text_chunks, embeddings,db_dir)


# Create a placeholder for the response
    response_placeholder = st.empty()

    non_streaming_llm = setup_llm(response_placeholder, streaming=False)
    streaming_llm = setup_llm(response_placeholder, streaming=True)
    

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are a highly knowledgeable conversational AI language model assistant. Your task is to generate three
        different versions of the given user question to retrieve the most relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines. Maintain a professional and friendly tone.
        Original question: {question}"""
    )

    retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    non_streaming_llm,  # Use non-streaming LLM for generating alternative questions
    prompt=QUERY_PROMPT
)

    template = """
        You are academic chatbot needs to answer the questions with repect to the context and question.
        If the query is not realted to the context answer on your own but it need to be accuarte
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = MessageHistoryChain(retriever, streaming_llm, non_streaming_llm, prompt, get_session_history("session"))

    st.write("Ready to Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_message = chain.welcome_message()
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input("Your response"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()
            response = chain.invoke(
                inputs={"user_answer": user_prompt},
                response_container=response_container,
                memory=get_session_history("session"),
                retriever=retriever
            )

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.input = ""
else:
    st.write("No data Loaded")


