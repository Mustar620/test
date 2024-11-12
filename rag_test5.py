import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RemoteRunnable
from langchain_core.output_parsers import StrOutputParser

def get_conversation_history():
    history = ""
    for msg in st.session_state["messages"]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return history

def main():
    st.set_page_config(
        page_title="Chat Bot",
        page_icon=":books:"
    )

    st.title("동서울대학교 chat bot:books:")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def print_history():
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def add_history(role, content):
        st.session_state["messages"].append({"role": role, "content": content})

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "안녕하세요! 궁금하신 것이 있으면 언제든 물어봐주세요!"
        }]

    print_history()

    if user_input := st.chat_input("메시지를 입력해 주세요"):
        add_history("user", user_input)
        st.chat_message("user").write(f"{user_input}")
        with st.chat_message("assistant"):

            llm = RemoteRunnable("https://mite-devoted-leech.ngrok-free.app/llm/")
            chat_container = st.empty()

            prompt_template = """
            당신은 동서울대학교 컴퓨터소프트웨어과 안내 AI 입니다.
            이전 대화를 바탕으로 질문에 맞는 답변을 50단어 보다 적게 해주세요.
            Conversation History:
            {history}
            Question: {question}
            Answer:
            """
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            conversation_history = get_conversation_history()

            chain = (
                {"question": user_input, "history": conversation_history}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = chain.stream(user_input)
            chunks = []
            for chunk in answer:
                chunks.append(chunk)
            chat_container.markdown("".join(chunks))
            add_history("ai", "".join(chunks))

if __name__ == '__main__':
    main()
