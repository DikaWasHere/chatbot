"""
Streamlit Web Interface untuk Chatbot RAG
Interface web sederhana untuk berinteraksi dengan chatbot
"""

import streamlit as st
import os
from chatbot_rag import ChatbotRAG


def initialize_chatbot():
    """
    Inisialisasi chatbot dan simpan di session state
    """
    if 'chatbot' not in st.session_state:
        with st.spinner('Menginisialisasi chatbot...'):
            chatbot = ChatbotRAG(use_openai=False)
            if chatbot.setup():
                st.session_state.chatbot = chatbot
                st.session_state.setup_complete = True
                st.success('Chatbot berhasil diinisialisasi!')
            else:
                st.error('Gagal menginisialisasi chatbot')
                st.session_state.setup_complete = False
    
    return st.session_state.get('setup_complete', False)


def initialize_chat_history():
    """
    Inisialisasi chat history
    """
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def add_to_chat_history(role: str, message: str):
    """
    Tambahkan pesan ke chat history
    
    Args:
        role: 'user' atau 'assistant'
        message: Isi pesan
    """
    st.session_state.chat_history.append({
        'role': role,
        'message': message
    })


def display_chat_history():
    """
    Tampilkan chat history
    """
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            with st.chat_message("user"):
                st.write(chat['message'])
        else:
            with st.chat_message("assistant"):
                st.write(chat['message'])


def main():
    """
    Main Streamlit app
    """
    # Page config
    st.set_page_config(
        page_title="Chatbot RAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🤖 Chatbot dengan RAG (Retrieval-Augmented Generation)")
    st.markdown("""
    Selamat datang di Chatbot RAG! Chatbot ini menggunakan teknologi Retrieval-Augmented Generation 
    untuk memberikan jawaban berdasarkan knowledge base yang tersedia.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Informasi")
        st.markdown("""
        **Knowledge Base meliputi:**
        - Python Programming
        - Artificial Intelligence & Machine Learning
        - Web Development dengan Python
        - Data Science dengan Python
        
        **Teknologi yang digunakan:**
        - LangChain
        - ChromaDB (Vector Database)
        - HuggingFace Embeddings
        - Streamlit (UI)
        """)
        
        st.header("💡 Tips Penggunaan")
        st.markdown("""
        - Tanyakan tentang topik yang tersedia di knowledge base
        - Gunakan bahasa Indonesia atau Inggris
        - Pertanyaan yang lebih spesifik akan memberikan hasil yang lebih baik
        
        **Contoh pertanyaan:**
        - "Apa itu Python?"
        - "Bagaimana cara belajar machine learning?"
        - "Framework apa saja untuk web development Python?"
        """)
        
        # Reset chat button
        if st.button("🗑️ Reset Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Initialize components
    initialize_chat_history()
    
    # Check if chatbot is initialized
    if not initialize_chatbot():
        st.stop()
    
    st.header("Ajukan Pertanyaan")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Tanyakan sesuatu..."):
        # Add user message to chat history
        add_to_chat_history("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    st.write(response)
                    add_to_chat_history("assistant", response)
                except Exception as e:
                    error_message = f"Terjadi error: {str(e)}"
                    st.error(error_message)
                    add_to_chat_history("assistant", error_message)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Dibuat dari andika untuk reina <3
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()