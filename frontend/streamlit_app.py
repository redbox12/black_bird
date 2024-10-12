import streamlit as st
import time
import requests

st.set_page_config(
    page_title=" Search chat",
    page_icon="🔍",
)

URL = "http://backend:80/question/"
QUESTION = "question"


# Streamed response emulator
def response(request):
    if request == "" or request== " ":
        yield "Введите валидное значение"

    data = {
        QUESTION: request,
    }

    response = requests.post(URL, json=data)
    if response.status_code == 200:
    # Если запрос успешен, распечатайте ответ
        data = response.json()
        for word in data.split():
            yield word + " "
            time.sleep(0.05) 
    
    else:
    # Если что-то пошло не так, распечатайте код состояния
        yield "Request failed with status {response.status_code}"
   


st.title("🔍 Search chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Что вы хотите уточнить?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})