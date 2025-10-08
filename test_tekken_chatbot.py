from chatbot_rag import ChatbotRAG

# Test chatbot dengan dokumen Tekken baru
def test_tekken_chatbot():
    print("=== Testing Chatbot dengan Dokumen Tekken ===")
    
    chatbot = ChatbotRAG(use_openai=False)
    
    if chatbot.setup():
        questions = [
            'Siapa itu Reina Mishima?',
            'Apa asal usul Reina?',
            'Apa kemampuan khusus Reina?',
            'Bagaimana hubungan Reina dengan keluarga Mishima?',
            'Apa itu Purple Lightning?',
            'Ceritakan tentang Devil Gene dalam keluarga Mishima'
        ]
        
        for q in questions:
            print(f'\n❓ Q: {q}')
            response = chatbot.chat(q)
            print(f'🤖 A: {response[:400]}...')
            print('-' * 80)
    else:
        print("❌ Gagal setup chatbot")

if __name__ == "__main__":
    test_tekken_chatbot()