import os

try:
    from langchain_huggingface import HuggingFaceEndpoint
except ImportError as exc:  # prettier error message when dependency is missing
    raise ImportError(
        "Package 'langchain-huggingface' belum terpasang. Jalankan `pip install langchain-huggingface langchain python-dotenv` terlebih dahulu."
    ) from exc

from dotenv import load_dotenv

# Load token dari file .env
load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    raise EnvironmentError(
        "Variabel lingkungan HUGGINGFACEHUB_API_TOKEN tidak ditemukan. Simpan token Hugging Face Anda ke dalam file `.env` seperti `HUGGINGFACEHUB_API_TOKEN=hf_xxx`."
    )

# Inisialisasi LLM dengan model tertentu
llm = HuggingFaceEndpoint(
    repo_id="distilgpt2",   # pilih model yang support text-generation
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

prompt = "Tuliskan pantun tentang belajar Python."
response = llm.invoke(prompt)

print(response)
