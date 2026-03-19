import streamlit as st
import requests
import uuid
import time

BACKEND_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Research AI Assistant 🤖", layout="wide")



# SESSION INIT
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "llm_id" not in st.session_state:
    st.session_state.llm_id = "llama-3.1-8b-instant"



# HELPERS
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "pdf_uploaded": False
    }
    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id):
    st.session_state.chats.pop(chat_id, None)
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = None
    st.rerun()


def send_message(chat_id, llm_id, messages, allow_search=False):
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "chat_id": chat_id,
                "llm_id": llm_id,
                "allow_search": allow_search,
                "messages": messages
            },
            timeout=60
        )

        return response.json() if response.status_code == 200 else {
            "answer": f"Error {response.status_code}: {response.text}",
            "source": "backend",
            "confidence": 0.0
        }

    except Exception as e:
        return {
            "answer": f"⚠️ Connection error: {str(e)}",
            "source": "client",
            "confidence": 0.0
        }


def upload_pdf(chat_id, file):
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        data = {"chat_id": chat_id}

        response = requests.post(
            f"{BACKEND_URL}/upload",
            files=files,
            data=data,
            timeout=120
        )

        if response.status_code == 200:
            return True

        return False

    except Exception as e:
        st.error(str(e))
        return False


def stream_text(text, delay=0.01):
    placeholder = st.empty()
    output = ""

    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(delay)

    return output



# SIDEBAR
st.sidebar.title("💬 Chats")

st.sidebar.selectbox(
    "Select LLM",
    [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ],
    key="llm_id"
)

allow_search = st.sidebar.toggle("🌐 Web Search", value=False)

if st.sidebar.button("➕ New Chat"):
    create_new_chat()

# Chat list
to_delete = None

for cid, chat in st.session_state.chats.items():
    col1, col2 = st.sidebar.columns([0.8, 0.2])

    if col1.button(chat["title"], key=cid):
        st.session_state.current_chat_id = cid

    if col2.button("❌", key=f"del_{cid}"):
        to_delete = cid

if to_delete:
    delete_chat(to_delete)


# MAIN
if st.session_state.current_chat_id is None:
    st.title("📄 Research Intelligence System")
    st.markdown("### 🚀 Upload papers & chat with them")
    st.stop()

chat_id = st.session_state.current_chat_id
chat_data = st.session_state.chats[chat_id]

st.title(chat_data["title"])



# PDF UPLOAD
if not chat_data["pdf_uploaded"]:
    file = st.file_uploader("📄 Upload Research Paper (PDF)", type=["pdf"])

    if file:
        with st.spinner("📥 Uploading & Processing..."):
            success = upload_pdf(chat_id, file)

        if success:
            chat_data["pdf_uploaded"] = True
            st.success("✅ PDF uploaded successfully! You can now chat.")
        else:
            st.error("❌ Upload failed")

    st.stop()



# CHAT DISPLAY
for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# INPUT
user_input = st.chat_input("Ask about your research paper...")

if user_input:
    # Update title
    if len(chat_data["messages"]) == 0:
        chat_data["title"] = user_input[:40]

    # Add user message
    chat_data["messages"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):

            response = send_message(
                chat_id=chat_id,
                llm_id=st.session_state.llm_id,
                messages=chat_data["messages"],
                allow_search=allow_search
            )

        answer = response.get("answer", "No response")

        # 🎯 Source Badge
        source = response.get("source", "unknown")
        confidence = response.get("confidence", 0)

        badge = f"📌 **Source:** `{source}` | 🎯 Confidence: `{round(confidence, 2)}`"
        st.markdown(badge)

        # ✨ Streaming effect
        final_text = stream_text(answer)

    # Save response
    chat_data["messages"].append({
        "role": "assistant",
        "content": final_text
    })