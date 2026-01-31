import google.generativeai as genai
import streamlit as st

st.title("The Musketeers Bot")

genai.configure(api_key=st.secrets["API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#React to user input
if prompt := st.chat_input("Ask anything", accept_file=True):

    user_text = prompt.text if prompt.text else ""

    #Display user message in chat message container
    with st.chat_message("user"):
        if user_text:
            st.markdown(user_text)
        if prompt.files:
            for f in prompt.files:
                st.caption(f"📎 Attached: {f.name}")

    #Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_text})

    with st.chat_message("assistant"):
        history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in st.session_state.messages[:-1]
        ]
        chat = model.start_chat(history=history)
        
        # Use stream=True for a better UX
        response_stream = chat.send_message(prompt, stream=True)
        
        # Stream the response to the UI
        full_response = st.write_stream(res.text for res in response_stream)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})