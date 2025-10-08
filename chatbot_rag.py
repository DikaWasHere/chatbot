"""
Chatbot dengan LangChain dan RAG
Implementasi chatbot yang menggunakan Retrieval-Augmented Generation
"""

import os
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from rag_system import RAGSystem


class ChatbotRAG:
    def __init__(self, openai_api_key: Optional[str] = None, use_openai: bool = False):
        """
        Inisialisasi Chatbot dengan RAG
        
        Args:
            openai_api_key: API key untuk OpenAI (opsional)
            use_openai: Apakah menggunakan OpenAI atau model lokal
        """
        self.use_openai = use_openai
        
        # Setup RAG system
        self.rag_system = RAGSystem()
        
        # Setup LLM
        if use_openai and openai_api_key:
            self.llm = ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            print("Menggunakan OpenAI GPT-3.5-turbo")
        else:
            # Untuk demo, kita akan menggunakan model sederhana
            # Dalam implementasi nyata, Anda bisa menggunakan model lokal seperti Ollama
            print("Mode demo - menggunakan template response sederhana")
            self.llm = None
        
        # Setup retriever
        self.retriever = None
        
        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan.

Konteks:
{context}

Instruksi:
1. Jawab pertanyaan berdasarkan konteks yang diberikan
2. Jika informasi tidak tersedia dalam konteks, katakan bahwa Anda tidak memiliki informasi tersebut
3. Berikan jawaban yang informatif dan membantu
4. Gunakan bahasa Indonesia yang baik dan benar
5. Jika relevan, berikan contoh atau penjelasan tambahan"""),
            ("human", "{question}")
        ])
    
    def setup(self) -> bool:
        """
        Setup chatbot dan RAG system
        
        Returns:
            True jika berhasil, False jika gagal
        """
        print("=== Setup Chatbot RAG ===")
        
        # Setup RAG system
        if not self.rag_system.setup_rag():
            print("Gagal setup RAG system")
            return False
        
        # Setup retriever
        self.retriever = self.rag_system.get_retriever(k=3)
        if not self.retriever:
            print("Gagal setup retriever")
            return False
        
        print("=== Chatbot RAG berhasil disetup ===")
        return True
    
    def format_docs(self, docs) -> str:
        """
        Format dokumen menjadi string context
        
        Args:
            docs: List of documents
            
        Returns:
            Formatted context string
        """
        return "\n\n".join([doc.page_content for doc in docs])
    
    def get_response_simple(self, question: str, context: str = "") -> str:
        """
        Generate response sederhana tanpa LLM (untuk demo)
        
        Args:
            question: Pertanyaan user
            context: Konteks dari RAG (optional, will be retrieved if empty)
            
        Returns:
            Response string
        """
        # Simple pattern matching responses for demo
        question_lower = question.lower()
        
        # Default retrieval if no specific context needed
        if not context:
            relevant_docs = self.rag_system.search_documents(question, k=3)
            if relevant_docs:
                context = self.format_docs(relevant_docs)
        
        if "python" in question_lower:
            if "apa itu" in question_lower or "pengertian" in question_lower:
                return f"""Python adalah bahasa pemrograman tingkat tinggi yang sangat populer. Berdasarkan konteks yang saya miliki:

{context[:500]}...

Python sangat cocok untuk pemula karena sintaksnya yang mudah dipahami dan memiliki banyak aplikasi seperti web development, data science, dan AI."""
            
            elif "keunggulan" in question_lower or "kelebihan" in question_lower:
                return f"""Keunggulan Python berdasarkan informasi yang saya miliki:

1. Sintaks yang mudah dipahami dan dipelajari
2. Dukungan library yang sangat luas
3. Cross-platform (dapat berjalan di berbagai sistem operasi)
4. Open source dan gratis
5. Komunitas yang besar dan aktif

Informasi lebih detail:
{context[:300]}..."""
        
        elif ("artificial intelligence" in question_lower or 
              ("ai" in question_lower and ("apa itu ai" in question_lower or "tentang ai" in question_lower or question_lower.strip() == "ai"))):
            return f"""Artificial Intelligence (AI) adalah teknologi yang memungkinkan mesin untuk meniru kecerdasan manusia. Berdasarkan konteks:

{context[:400]}...

AI memiliki berbagai aplikasi seperti NLP, computer vision, dan machine learning."""
        
        elif "web development" in question_lower:
            return f"""Python memiliki framework yang powerful untuk web development. Berdasarkan informasi:

{context[:400]}...

Framework populer termasuk Django, Flask, dan FastAPI."""
        
        elif "data science" in question_lower:
            return f"""Data Science menggunakan Python dengan berbagai library. Informasi:

{context[:400]}...

Library utama meliputi NumPy, Pandas, Matplotlib, dan Scikit-learn."""
        
        elif "esther" in question_lower:
            # Use more specific search for Esther
            relevant_docs = self.rag_system.search_documents("Reina Esther perasaan menyukai", k=3)
            if relevant_docs:
                context = self.format_docs(relevant_docs)
                # Look for the specific information about Esther
                for doc in relevant_docs:
                    if "esther" in doc.page_content.lower():
                        context = doc.page_content
                        break
            
            return f"""Berdasarkan informasi tentang Esther:

{context[:400]}...

Esther adalah seseorang yang disukai oleh Reina Mishima. Reina sudah lama menyimpan perasaan terhadap Esther."""
        
        elif "andika" in question_lower:
            # Use more specific search for Andika
            relevant_docs = self.rag_system.search_documents("Reina Andika crush hubungan", k=3)
            if relevant_docs:
                context = self.format_docs(relevant_docs)
                # Look for the specific information about Andika
                for doc in relevant_docs:
                    if "andika" in doc.page_content.lower():
                        context = doc.page_content
                        break
            
            return f"""Berdasarkan informasi tentang Andika:

{context[:400]}...

Andika adalah seseorang yang memiliki perasaan crush terhadap Reina Mishima yang sudah lama dipendam."""
        
        elif "siapa yang disukai" in question_lower or "yang disukai reina" in question_lower:
            # Use more specific search for Reina's crush
            relevant_docs = self.rag_system.search_documents("Reina menyukai esther perasaan", k=3)
            if relevant_docs:
                context = self.format_docs(relevant_docs)
                # Look for the specific information about who Reina likes
                for doc in relevant_docs:
                    if "esther" in doc.page_content.lower():
                        context = doc.page_content
                        break
            
            return f"""Berdasarkan informasi pribadi Reina:

{context[:400]}...

Reina menyukai seseorang bernama Esther yang sudah lama dia taksir."""
        
        elif "reina" in question_lower:
            if "siapa yang disukai" in question_lower or "yang disukai" in question_lower:
                # Handle "who does Reina like" within Reina block
                relevant_docs = self.rag_system.search_documents("Reina menyukai esther perasaan", k=3)
                if relevant_docs:
                    context = self.format_docs(relevant_docs)
                    for doc in relevant_docs:
                        if "esther" in doc.page_content.lower():
                            context = doc.page_content
                            break
                
                return f"""Berdasarkan informasi pribadi Reina:

{context[:400]}...

Reina menyukai seseorang bernama Esther yang sudah lama dia taksir."""
            
            elif "siapa" in question_lower or "apa itu" in question_lower:
                return f"""Reina Mishima adalah karakter baru dalam Tekken 8 dan anggota keluarga Mishima yang misterius. Berdasarkan informasi yang saya miliki:

{context[:500]}...

Reina adalah putri Heihachi Mishima dan memiliki kemampuan Purple Lightning yang unik."""
            
            elif "asal usul" in question_lower or "latar belakang" in question_lower:
                return f"""Asal usul Reina sangat menarik dalam lore Tekken. Berdasarkan informasi:

{context[:500]}...

Reina lahir dari hubungan rahasia Heihachi dan dibesarkan jauh dari konflik keluarga Mishima."""
            
            elif "kemampuan" in question_lower or "kekuatan" in question_lower:
                return f"""Kemampuan tempur Reina sangat unik dalam keluarga Mishima:

{context[:500]}...

Dia terkenal dengan julukan 'Purple Lightning' karena teknik listrik ungunya."""
            
            elif "hubungan" in question_lower or "keluarga" in question_lower:
                return f"""Hubungan Reina dengan keluarga Mishima sangat kompleks:

{context[:500]}...

Sebagai putri Heihachi, dia memiliki hubungan rumit dengan semua anggota keluarga."""
        
        elif "tekken" in question_lower:
            return f"""Tekken memiliki alur cerita yang sangat mendalam. Berdasarkan informasi:

{context[:400]}...

Serial ini terkenal dengan konflik keluarga Mishima yang berlangsung turun-temurun."""
        
        elif "purple lightning" in question_lower:
            return f"""Purple Lightning adalah kemampuan khas Reina Mishima:

{context[:400]}...

Ini adalah varian unik dari teknik listrik keluarga Mishima dengan warna ungu."""
        
        elif "devil gene" in question_lower:
            return f"""Devil Gene adalah elemen penting dalam lore Tekken:

{context[:400]}...

Gen setan ini diturunkan dalam keluarga Mishima dan memberikan kekuatan supernatural."""
        
        elif "mishima" in question_lower:
            return f"""Keluarga Mishima adalah inti dari cerita Tekken:

{context[:400]}...

Konflik keluarga ini telah mempengaruhi dunia selama beberapa generasi."""
        
        else:
            # Generic response
            return f"""Berdasarkan informasi yang saya miliki:

{context[:500]}...

Apakah ada aspek spesifik yang ingin Anda ketahui lebih lanjut?"""
    
    def get_response_openai(self, question: str) -> str:
        """
        Generate response menggunakan OpenAI
        
        Args:
            question: Pertanyaan user
            
        Returns:
            Response string
        """
        # Create RAG chain
        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Generate response
        response = rag_chain.invoke(question)
        return response
    
    def chat(self, question: str) -> str:
        """
        Main chat function
        
        Args:
            question: Pertanyaan user
            
        Returns:
            Response dari chatbot
        """
        if not self.retriever:
            return "Chatbot belum disetup. Silakan panggil setup() terlebih dahulu."
        
        try:
            # Generate response (retrieval will be handled inside get_response_simple for better context)
            if self.use_openai and self.llm:
                # For OpenAI, use default retrieval
                relevant_docs = self.rag_system.search_documents(question, k=3)
                if not relevant_docs:
                    return "Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan Anda."
                response = self.get_response_openai(question)
            else:
                # For simple mode, let get_response_simple handle specific retrieval
                response = self.get_response_simple(question, "")
            
            return response
            
        except Exception as e:
            return f"Terjadi error: {e}"
    
    def interactive_chat(self):
        """
        Mode chat interaktif
        """
        print("\n" + "="*50)
        print("🤖 Chatbot RAG - Siap membantu Anda!")
        print("Ketik 'quit' atau 'exit' untuk keluar")
        print("="*50 + "\n")
        
        while True:
            try:
                question = input("Anda: ").strip()
                
                if question.lower() in ['quit', 'exit', 'keluar']:
                    print("Terima kasih! Sampai jumpa! 👋")
                    break
                
                if not question:
                    continue
                
                print("\n🤖 Bot: ", end="")
                response = self.chat(question)
                print(response)
                print("\n" + "-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nTerima kasih! Sampai jumpa! 👋")
                break
            except Exception as e:
                print(f"\nTerjadi error: {e}")


def main():
    """
    Main function untuk test chatbot
    """
    # Inisialisasi chatbot (tanpa OpenAI untuk demo)
    chatbot = ChatbotRAG(use_openai=False)
    
    # Setup chatbot
    if not chatbot.setup():
        print("Gagal setup chatbot")
        return
    
    # Test beberapa pertanyaan
    test_questions = [
        "Apa itu Python?",
        "Apa keunggulan Python?",
        "Bagaimana cara belajar Python?",
        "Apa itu AI dan machine learning?"
    ]
    
    print("\n=== Test Chatbot ===")
    for question in test_questions:
        print(f"\nQ: {question}")
        print(f"A: {chatbot.chat(question)}")
        print("-" * 80)
    
    # Mode interaktif
    print("\n=== Mode Interaktif ===")
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()