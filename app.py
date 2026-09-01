https://chatgpt-ruq982wake3h8jmjo46gll.streamlit.app/

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="LangChain ChatGPT Clone", page_icon="💬", layout="centered")

# ----------------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------------
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------------------------------------------------
# Shared: session store for RunnableWithMessageHistory
# ----------------------------------------------------------------------------
@st.cache_resource
def get_store():
    return {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = get_store()
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def validate_api_key(key: str) -> tuple[bool, str]:
    """Make a tiny real request to confirm the key actually works."""
    if not key or not key.startswith("sk-"):
        return False, "That doesn't look like a valid OpenAI key (should start with 'sk-')."
    try:
        test_llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=key, max_tokens=1)
        test_llm.invoke("hi")
        return True, ""
    except Exception as e:
        return False, f"Key rejected by OpenAI: {e}"


@st.cache_resource(show_spinner=False)
def build_chain(_api_key: str, _model_name: str, _temperature: float, _system_prompt: str):
    llm = ChatOpenAI(
        model_name=_model_name,
        temperature=_temperature,
        openai_api_key=_api_key,
    )

    prompt = ChatPromptTemplate(
        input_variables=["content", "chat_history"],
        messages=[
            SystemMessage(content=_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{content}"),
        ],
    )

    base_chain = prompt | llm | StrOutputParser()

    chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="content",
        history_messages_key="chat_history",
    )
    return chain


# ============================================================================
# SCREEN 1 — API key gate
# ============================================================================
def render_gate():
    st.title("💬 LangChain ChatGPT Clone")
    st.write("Enter your OpenAI API key to continue.")

    with st.form("api_key_form"):
        key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your key is only kept in this browser session — never written to disk.",
        )
        submitted = st.form_submit_button("Continue →", use_container_width=True)

    if submitted:
        with st.spinner("Validating key..."):
            ok, error = validate_api_key(key_input)

        if ok:
            st.session_state.api_key = key_input
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error(error)

    st.caption("Don't have a key? Get one at platform.openai.com/api-keys")


# ============================================================================
# SCREEN 2 — Chat interface
# ============================================================================
def render_chat():
    with st.sidebar:
        st.header("⚙️ Settings")

        st.success("API key set ✓")
        if st.button("🔑 Change API key", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.api_key = ""
            build_chain.clear()
            st.rerun()

        model_name = st.selectbox(
            "Model",
            options=["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=0,
        )

        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)

        system_prompt = st.text_area(
            "System prompt",
            value="You are a chatbot having a conversation with a human.",
            height=90,
        )

        st.divider()

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            get_store().pop("default", None)
            st.rerun()

        st.caption("Session ID: `default`")

    st.title("💬 ChatGPT Clone")
    st.caption("Built with LangChain + Streamlit — conversation memory included.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        chain = build_chain(
            st.session_state.api_key, model_name, temperature, system_prompt
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke(
                        {"content": user_input},
                        config={"configurable": {"session_id": "default"}},
                    )
                except Exception as e:
                    response = f"⚠️ Error: {e}"
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ============================================================================
# Router
# ============================================================================
if not st.session_state.authenticated:
    render_gate()
else:
    render_chat()
