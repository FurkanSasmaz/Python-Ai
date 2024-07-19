# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:38:55 2024

@author: furkan-sasmaz
"""

from openai import OpenAI
from pymongo import MongoClient
import json
from datetime import datetime
import nltk
import string

nltk.download('punkt')

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Upper & lower case
    text = text.lower()
    # Remove unnecessary spaces
    text = ' '.join(text.split())
    return text

def get_next_id(collection):
    count = collection.count_documents({})
    return count + 1

def is_record_already_saved(question, collection):
    existing_record = collection.find_one({'question': question})
    return existing_record is not None

def save_vector_to_mongodb(question, answer, text_vector, collection):
    if is_record_already_saved(question, collection):
        print(f'Bu soru zaten MongoDB\'ye kaydedilmiş! ({question})')
    else:
        next_id = get_next_id(collection)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vector_data = {
            'question': question,
            'answer': answer,
            'id': next_id,
            'embedding_vector': text_vector,
            'uploadDate': timestamp
        }
        collection.insert_one(vector_data)
        print(f'Soru-cevap çifti başarıyla MongoDB\'ye kaydedildi! ({question})')

# Connect MongoDB
mongodb_client = MongoClient('mongodb://localhost:27017/')
db = mongodb_client['db-name']
collection = db['collection-name']

# Read JSON file and create embedding vectors and save to MongoDB
json_file_path = "responses2.json"
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    qa_pairs = json.load(json_file)

# Build the OpenAI client
openai_client = OpenAI(organization="org-key", api_key="api-key")

# Her soru-cevap çifti için embedding vektörü oluştur ve MongoDB'ye kaydet
for pair in qa_pairs:
    question = preprocess_text(pair['question'])
    answer = pair['answer']
    
    #Soruyu vektöre çevir
    response = openai_client.embeddings.create(
        model="embedding-model",
        input=[question]
    )
    embedding_vector = response.data[0].embedding

    # Vektörü MongoDB'ye kaydet
    save_vector_to_mongodb(question, answer, embedding_vector, collection)

# MongoDB bağlantısını kapat
mongodb_client.close()

