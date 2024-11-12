import streamlit as st
import tiktoken
from loguru import logger

from langchain_core.messages import ChatMessage
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

def get_conversation_history():
    """히스토리를 하나의 문자열로 결합하여 반환합니다."""
    history = ""
    for msg in st.session_state["messages"]:
        role = "User" if msg.role == "user" else "Assistant"
        history += f"{role}: {msg.content}\n"
    return history

def main():
    st.set_page_config(
        page_title="ChatBot",
        page_icon=":books:"
    )

    st.title("동서울대학교 chat bot :books:")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def print_history():
        for msg in st.session_state.messages:
            st.chat_message(msg.role).write(msg.content)

    def add_history(role, content):
        st.session_state.messages.append(ChatMessage(role=role, content=content))

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{
            "role": "assistant",
            "content": "안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요!"
        }]

    SIMPLE_PROMPT_TEMPLATE = """
    당신은 동서울대학교 컴퓨터소프트웨어과 안내 AI 입니다.
    질문에 간결하게 답변하세요.
    Conversation History:
    {history}
    Question: {question}
    Answer:
    """

    print_history()

    if user_input := st.chat_input("메시지를 입력해 주세요"):
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")
        with st.chat_message("assistant"):

            llm = RemoteRunnable("https://mite-devoted-leech.ngrok-free.app/llm/")
            chat_container = st.empty()

            prompt = ChatPromptTemplate.from_template(SIMPLE_PROMPT_TEMPLATE)
            conversation_history = get_conversation_history()

            chain = (
                {
                    "history": lambda _: conversation_history,  # 대화 히스토리 전달
                    "question": user_input                       # 사용자 질문 전달
                }
                | prompt
                | llm
            )

            answer = chain.stream(user_input)
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
            chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))

if __name__ == '__main__':
    main()
