from openai import OpenAI
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #Upper & lower case 
    text = text.lower()
    # Gereksiz beyaz boşlukları kaldırma
    text = ' '.join(text.split())
    return text

def find_most_similar_answer(question, collection):
    # Kullanıcıdan gelen soruyu ön işlemden geçir
    processed_question = preprocess_text(question)
    
    # Soruyu embedding vektörüne dönüştür
    response = openai_client.embeddings.create(
        model="embedding-model", 
        input=[processed_question]
    )
    question_embedding = np.array(response.data[0].embedding).reshape(1, -1)
    
    # MongoDB'deki tüm embedding vektörlerini ve soru-cevap çiftlerini çek
    all_vectors = list(collection.find({}, {'embedding_vector': 1, 'question': 1, 'answer': 1, '_id': 0}))
    
    # Tüm vektörlerle benzerlik hesapla
    similarities = []
    for vector_data in all_vectors:
        db_vector = np.array(vector_data['embedding_vector']).reshape(1, -1)
        similarity = cosine_similarity(question_embedding, db_vector)[0][0]
        similarities.append((similarity, vector_data['question'], vector_data['answer']))
    
    # En yüksek benzerliğe sahip olanı seç
    most_similar = max(similarities, key=lambda x: x[0])
    return most_similar[1], most_similar[2], most_similar[0]

# OpenAI istemcisini oluştur
openai_client = OpenAI(organization="org-key", api_key="api-key")

# MongoDB'ye bağlan
mongodb_client = MongoClient('mongodb://localhost:27017/')
db = mongodb_client['db-name']
collection = db['collection-name']

# Kullanıcıdan gelen soru
user_question = "1.osmanlı padişahı kimdir?"

# En benzer cevabı bul
most_similar_question, answer, similarity = find_most_similar_answer(user_question, collection)

if similarity > 0.9: #optional
    print(f"Sorulan Soru: {user_question}")
    print(f"En Benzer Soru: {most_similar_question}")
    print(f"Cevap: {answer}")
    print(f"Benzerlik Skoru: {similarity}")
else:
    print("Maalesef seni tam olarak anlayamadım. Sormuş olduğun soruyu daha farklı bir şekilde sormayı dene.")
    print(f"En Benzer Soru: {most_similar_question}")
    print(f"Benzerlik Skoru: {similarity}")
    
# MongoDB bağlantısını kapat
mongodb_client.close()



