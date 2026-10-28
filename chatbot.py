from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

st.title("🤖 Ask-Buddy QnA ChatBot")
st.markdown("My chatbot with langchain and Google gemini")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

if "messages" not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    role = messages["role"]
    content = messages["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("ASK ANYTHING HERE")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    res = llm.invoke(query)
    answer = res.content

    if isinstance(answer, list):
        answer = "\n".join(
            block["text"]
            for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
