import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

def get_text():
    doc_list = []

    base_path = os.path.dirname(__file__)
    pdf_path1 = os.path.join(base_path, "data", "컴퓨터소프트웨어학과.pdf")
    pdf_path2 = os.path.join(base_path, "data", "indata_kor.pdf")
    
    loader = PyPDFLoader(pdf_path1)
    documents = loader.load()
    doc_list.extend(documents)
    
    loader = PyPDFLoader(pdf_path2)
    documents = loader.load()
    doc_list.extend(documents)
    
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_history():
    history = ""
    for msg in st.session_state["messages"]:
        role = "User" if msg.role == "user" else "Assistant"
        history += f"{role}: {msg.content}\n"
    return history

def main():
    st.set_page_config(
        page_title="RAG+PEFT",
        page_icon=":books:"
    )

    st.title("동서울대학교 chat bot:books:")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "store" not in st.session_state:
        st.session_state["store"] = dict()

    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(role, content):
        st.session_state.messages.append({"role": role, "content": content})

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    files_text = get_text()
    text_chunks = get_text_chunks(files_text)
    vectorstore = get_vectorstore(text_chunks)
    retriever = vectorstore.as_retriever(search_type='mmr', verbose=True)
    st.session_state['retriever'] = retriever
    st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{
            "role": "assistant",
            "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"
        }]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    RAG_PROMPT_TEMPLATE = """
    당신은 동서울대학교 컴퓨터소프트웨어과 안내 AI 입니다.
    검색된 문맥을 사용하여 질문에 맞는 답변을 50단어 보다 적게 해주세요
    Conversation History:
    {history}
    Question: {question}
    Context: {context}
    Answer:
    """

    print_history()

    if user_input := st.chat_input("메시지를 입력해 주세요"):
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")
        with st.chat_message("assistant"):

            llm = ChatOllama(model="Llama3_Bllo_8b_q4_k_m:latest")
            chat_container = st.empty()

            if st.session_state.processComplete:
                prompt1 = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

                conversation_history = get_conversation_history()

                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                        "history": RunnablePassthrough(lambda _: conversation_history)
                    }
                    | prompt1
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.stream(user_input)
                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                chat_container.markdown("".join(chunks))
                add_history("ai", "".join(chunks))
            
            else:
                prompt2 = ChatPromptTemplate.from_template(
                    "다음의 질문에 간결하게 답변해 주세요:\n{input}"
                )
                chain = prompt2 | llm | StrOutputParser()
                answer = chain.stream(user_input)
                chunks = []
                for chunk in answer:
                    chunks.append(chunk)
                chat_container.markdown("".join(chunks))
                add_history("ai", "".join(chunks))

if __name__ == '__main__':
    main()
