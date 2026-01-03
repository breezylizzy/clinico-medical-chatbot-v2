import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.retriever import get_base_retriever
from src.chain import create_rag_chain
from src.config import OPENAI_API_KEY

def main():
    st.set_page_config(
        page_title="CLINICO Medical Assistant",
        page_icon="🩺",
        layout="centered",
    )

    st.markdown("""
    <style>


        /* =====================================================
        CHAT BUBBLES 
        ===================================================== */

        /* USER */
        .stChatMessage.user {
            background: #0d3b66 !important;         /* biru sedang */
            color: #e5f0ff !important;              /* biru muda */
            border-radius: 18px;
            padding: 12px 16px;
            margin: 10px 0;
            border: 1px solid rgba(229, 240, 255, 0.2) !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        }

        /* ASSISTANT */
        .stChatMessage.assistant {
            background: rgba(255,255,255,0.06) !important;   /* putih tipis */
            color: #e5f0ff !important;
            border-radius: 18px;
            padding: 12px 16px;
            margin: 10px 0;
            border: 1px solid rgba(255,255,255,0.1) !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            backdrop-filter: blur(4px);
        }


        /* =====================================================
        INPUT BOX 
        ===================================================== */
        .stChatInput > div > div {
            background: #0d3b66 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(229, 240, 255, 0.2) !important;
            box-shadow: 0 3px 6px rgba(0,0,0,0.35) !important;
            color: #e5f0ff !important;
        }

        input, textarea {
            background: transparent !important;
            color: #e5f0ff !important;
        }

        input::placeholder, textarea::placeholder {
            color: rgba(229, 240, 255, 0.4) !important;
        }


         /* =====================================================
        BACKGROUND GRADIENT VERTICAL (Hitam → Biru → Hitam)
        ===================================================== */
        .stApp {
            background: linear-gradient(
                to bottom,
                #000000 0%,
                #0a1a2a 40%,
                #0d3b66 60%,
                #000000 100%
            ) !important;
            background-attachment: fixed !important;
        }
        </style>
    """, unsafe_allow_html=True)


    st.title("🩺 CLINICO - Medical Information Assistant")
    st.markdown("""
    CLINICO helps provide safe and informative health-related answers.
    It does not offer medical diagnoses or definitive prescriptions.  
    """)

    try:
        chat_model = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, 
            model="gpt-4o-mini", 
            temperature=0.0
        )

        rag_chain_with_memory = create_rag_chain(chat_model=chat_model)
        
        if "session_id" not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
            
        if "messages" not in st.session_state:
            st.session_state.messages = [
                AIMessage(content="Hello! I am CLINICO. How can I help you today?")
            ]

    except Exception as e:
        st.error(f"Gagal menginisialisasi komponen RAG. Pastikan semua file dependensi sudah benar dan kunci API tersedia: {e}")
        return

    for message in st.session_state.messages:
        avatar = "👤" if isinstance(message, HumanMessage) else "🤖"
        with st.chat_message(message.type, avatar=avatar):
            st.markdown(message.content)
    
    if prompt := st.chat_input("“Ask something about health...”"):
        
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.spinner("CLINICO is looking for information..."):
            try:
                response = rag_chain_with_memory.invoke(
                    {"input": prompt},
                    config={
                        "configurable": {"session_id": st.session_state.session_id}
                    }
                )

                ai_response = response.get(
                    "answer",
                    "Sorry, I couldn’t find any related information."
                )
                
                NO_SOURCE_MESSAGE = "<<< NO_RELEVANT_SOURCES >>>"

                if NO_SOURCE_MESSAGE in ai_response:
                    ai_response = "I could not find any relevant sources."
                elif "context" in response and response["context"]:
                    
                    sources = {
                        os.path.basename(doc.metadata.get("source")) 
                        for doc in response["context"] 
                        if doc.metadata.get("source") 
                    }
                    
                    if sources:
                        source_text = "\n\n**References used:**\n" + "\n".join(f"* {s}" for s in sources)
                        ai_response += source_text
            except Exception as e:
                ai_response = f"An error occurred while processing your request: {e}"

        st.session_state.messages.append(AIMessage(content=ai_response))
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(ai_response)

if __name__ == "__main__":
    main()
