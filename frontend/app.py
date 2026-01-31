import os

import requests
import streamlit as st


st.set_page_config(page_title="LangChain FastAPI + Streamlit", page_icon="🦜")

st.title("LangChain FastAPI + Streamlit")
st.write("Send a message to the FastAPI backend and view the response.")

api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

with st.form("chat_form"):
    message = st.text_input("Message", placeholder="Ask something...")
    submitted = st.form_submit_button("Send")

if submitted:
    if not message.strip():
        st.warning("Please enter a message before sending.")
    else:
        try:
            response = requests.post(
                f"{api_base_url}/chat", json={"message": message}, timeout=10
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        else:
            st.success("Response received")
            st.write(data.get("response", "(no response)"))
