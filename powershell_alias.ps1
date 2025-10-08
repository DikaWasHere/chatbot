# Tambahkan ke PowerShell Profile: $PROFILE
# Untuk membuat alias permanent

function Start-TekkenChatbot {
    Set-Location "d:\semester 7\magang belajar"
    & "C:\Users\andik\anaconda3\python.exe" -m streamlit run streamlit_app.py
}

# Alias singkat
Set-Alias tekken Start-TekkenChatbot
Set-Alias chatbot Start-TekkenChatbot