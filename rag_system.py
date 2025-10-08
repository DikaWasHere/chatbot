"""
RAG System Implementation
Sistem untuk load dokumen, split text, create embeddings, dan setup vector database
"""

import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

class RAGSystem:
    def __init__(self, documents_path: str = "documents", persist_directory: str = "chroma_db"):
        """
        Inisialisasi RAG System
        
        Args:
            documents_path: Path ke folder yang berisi dokumen
            persist_directory: Path untuk menyimpan vector database
        """
        self.documents_path = documents_path
        self.persist_directory = persist_directory
        
        # Inisialisasi embeddings menggunakan model gratis dari HuggingFace
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Inisialisasi text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.vectorstore = None
    
    def load_documents(self) -> List[Document]:
        """
        Load semua dokumen dari folder documents
        
        Returns:
            List of Document objects
        """
        try:
            # Load semua file .txt dari folder documents
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
            
            print(f"Berhasil memuat {len(documents)} dokumen")
            return documents
            
        except Exception as e:
            print(f"Error saat memuat dokumen: {e}")
            return []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split dokumen menjadi chunks yang lebih kecil
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of split Document objects
        """
        try:
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Dokumen dibagi menjadi {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            print(f"Error saat split dokumen: {e}")
            return []
    
    def create_vectorstore(self, documents: List[Document]) -> bool:
        """
        Buat vector store dari dokumen
        
        Args:
            documents: List of Document objects
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            # Buat vector store menggunakan Chroma
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"Vector store berhasil dibuat dengan {len(documents)} dokumen")
            return True
            
        except Exception as e:
            print(f"Error saat membuat vector store: {e}")
            return False
    
    def load_existing_vectorstore(self) -> bool:
        """
        Load vector store yang sudah ada
        
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print("Vector store yang sudah ada berhasil dimuat")
                return True
            else:
                print("Vector store belum ada, perlu dibuat terlebih dahulu")
                return False
                
        except Exception as e:
            print(f"Error saat memuat vector store: {e}")
            return False
    
    def setup_rag(self) -> bool:
        """
        Setup lengkap RAG system
        
        Returns:
            True jika berhasil, False jika gagal
        """
        print("=== Setup RAG System ===")
        
        # Coba load vector store yang sudah ada
        if self.load_existing_vectorstore():
            return True
        
        # Jika belum ada, buat dari awal
        print("Membuat vector store baru...")
        
        # 1. Load dokumen
        documents = self.load_documents()
        if not documents:
            return False
        
        # 2. Split dokumen
        split_docs = self.split_documents(documents)
        if not split_docs:
            return False
        
        # 3. Buat vector store
        success = self.create_vectorstore(split_docs)
        
        if success:
            print("=== RAG System berhasil disetup ===")
        
        return success
    
    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Cari dokumen yang relevan dengan query
        
        Args:
            query: Pertanyaan atau query
            k: Jumlah dokumen yang akan dikembalikan
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            print("Vector store belum diinisialisasi")
            return []
        
        try:
            relevant_docs = self.vectorstore.similarity_search(query, k=k)
            print(f"Ditemukan {len(relevant_docs)} dokumen relevan")
            return relevant_docs
            
        except Exception as e:
            print(f"Error saat mencari dokumen: {e}")
            return []
    
    def get_retriever(self, k: int = 3):
        """
        Dapatkan retriever untuk RAG chain
        
        Args:
            k: Jumlah dokumen yang akan dikembalikan
            
        Returns:
            Vectorstore retriever
        """
        if not self.vectorstore:
            print("Vector store belum diinisialisasi")
            return None
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


def main():
    """
    Test RAG system
    """
    # Inisialisasi RAG system
    rag = RAGSystem()
    
    # Setup RAG
    if rag.setup_rag():
        # Test search
        query = "Apa itu Python?"
        relevant_docs = rag.search_documents(query)
        
        print(f"\nQuery: {query}")
        print("Dokumen relevan:")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\n{i}. {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
    
    else:
        print("Gagal setup RAG system")


if __name__ == "__main__":
    main()