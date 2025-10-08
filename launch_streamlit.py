#!/usr/bin/env python3
"""
Simple launcher untuk Streamlit app
"""
import subprocess
import sys
import os

def run_streamlit():
    """Jalankan Streamlit app"""
    try:
        # Change to the correct directory
        os.chdir(r"d:\semester 7\magang belajar")
        
        # Use conda python executable
        python_exe = r"C:\Users\andik\anaconda3\python.exe"
        
        print("🚀 Starting Tekken Chatbot...")
        print("🌐 Streamlit akan terbuka di browser")
        print("⚡ Menggunakan Conda Python environment")
        print("-" * 50)
        
        # Run streamlit dengan conda python
        subprocess.run([python_exe, "-m", "streamlit", "run", "streamlit_app.py"])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    run_streamlit()