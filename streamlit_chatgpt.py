import openai
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import streamlit as st

# Page configuration
st.set_page_config(page_title="Your Custom Assistant!", page_icon="🤖")

# Initialize Session State
if "api_key" not in st.session_state:
  st.session_state.api_key = ""
if "authenticated" not in st.session_state:
  st.session_state.authenticated = False
if "messages" not in st.session_state:
  st.session_state.messages = []


def validate_api_key(key: str) -> bool:
  """Test if the provided OpenAI API key is valid."""
  try:
    client = openai.OpenAI(api_key=key)
    client.models.list()
    return True
  except Exception:
    return False


# --- GATE 1: API KEY INPUT SCREEN ---
if not st.session_state.authenticated:
  st.title("🔑 Welcome! Enter Your API Key")
  st.write(
      "Please enter a valid OpenAI API key below to access the assistant."
  )

  api_key_input = st.text_input(
      "OpenAI API Key",
      type="password",
      placeholder="sk-...",
      help="Your key is stored only in session state and not saved.",
  )

  if st.button("Submit & Proceed"):
    if api_key_input:
      with st.spinner("Validating API Key..."):
        if validate_api_key(api_key_input):
          st.session_state.api_key = api_key_input
          st.session_state.authenticated = True
          st.success("API key validated successfully!")
          st.rerun()
        else:
          st.error(
              "Invalid API Key. Please check your key and try again."
          )
    else:
      st.warning("Please enter an API key.")

# --- GATE 2: MAIN CHAT INTERFACE ---
else:
  st.subheader("Your Custom ChatGPT 🤖")

  # Initialize ChatOpenAI using the user's validated key
  chat = ChatOpenAI(
      model_name="gpt-3.5-turbo",
      temperature=0.5,
      openai_api_key=st.session_state.api_key,
  )

  # Sidebar Controls
  with st.sidebar:
    st.write("### Settings")
    system_message = st.text_input(label="System Role")

    if system_message:
      if not any(
          isinstance(x, SystemMessage) for x in st.session_state.messages
      ):
        st.session_state.messages.append(SystemMessage(content=system_message))

    st.divider()
    if st.button("Change API Key"):
      st.session_state.authenticated = False
      st.session_state.api_key = ""
      st.session_state.messages = []
      st.rerun()

  # System Message Fallback
  if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], SystemMessage):
      st.session_state.messages.insert(
          0, SystemMessage(content="You are a helpful Assistant")
      )

  # Render Chat History
  for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
      with st.chat_message("human"):
        st.write(msg.content)
    elif isinstance(msg, AIMessage):
      with st.chat_message("ai"):
        st.write(msg.content)

  # Chat Input Widget
  user_prompt = st.chat_input("Type your message here...")

  if user_prompt:
    st.session_state.messages.append(HumanMessage(content=user_prompt))

    with st.chat_message("human"):
      st.write(user_prompt)

    with st.chat_message("ai"):
      with st.spinner("Working on your Request..."):
        try:
          response = chat.invoke(st.session_state.messages)
          st.write(response.content)
          st.session_state.messages.append(AIMessage(content=response.content))
        except Exception as e:
          st.error(f"Error processing request: {e}")