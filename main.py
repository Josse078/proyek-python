import os
import json
from difflib import get_close_matches
from transformers import BertTokenizer, BertForQuestionAnswering, GPT2LMHeadModel, GPT2Tokenizer
import torch
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy  
from sentence_transformers import SentenceTransformer, util  
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re
import textwrap
from transformers import AdamW

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = BertForQuestionAnswering.from_pretrained("indobenchmark/indobert-base-p1")
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
nlp = spacy.load("en_core_web_sm")
def generate_answer_gpt2(question):
    input_ids = gpt2_tokenizer.encode(question, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    answer = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

def load_knowledge_base(file_path:str) -> dict:
    with open(file_path, 'r') as file:
        data: dict = json.load(file)
    return data

def find_similar_question(user_question, dataset):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform([user_question])
    most_similar_question = None
    max_similarity = 0
    for entry in dataset:
        entry_vector = vectorizer.transform([entry['question']])
        similarity = cosine_similarity(question_vectors, entry_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_question = entry['question']
    return most_similar_question

def save_knowledge_base(file_path:str, data:dict):
    with open(file_path, 'w') as file: 
        json.dump(data, file, indent=2)

def find_best_match(user_question:str, questions:list[str]) -> str | None:
    matches:list = get_close_matches(user_question, questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_answer_for(question: str, knowledge_base:dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]

def answer_question(question, dataset):
    similar_question = find_similar_question(question, dataset)
    user_input = preprocess_text(question)

    if similar_question is not None:
        for entry in dataset:
            if similar_question.lower() == entry['question'].lower() or similar_question.lower() == entry['answer'].lower():
                return entry['answer']

    google_results = list(search(question, num_results=5))

    search_links = []
    for result in google_results:
        search_links.append(result)

    if search_links:
        return f"I couldn't find a similar question in my knowledge base. Here are some relevant search results on Google:\n{', '.join(search_links)}"
    else:
        
        gpt2_answer = generate_answer_gpt2(question)
        return f"GPT-2 says: {gpt2_answer}"

def load_qa_dataset(file_path):
    dataset = []
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            question = row['question']
            answer = row['answer']
            dataset.append({'question': question, 'answer': answer})
            dataset.append({'question': answer, 'answer': question})  
    return dataset

def chat_bot(dataset, knowledge_base_path):
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break

       
        answer = answer_question(user_input, dataset)

        
        edit_response = input(f'Bot: {answer}\nIs this response accurate? (yes/no): ').lower()

        if edit_response == 'no':
            edited_response = input('Please provide the corrected response: ')
           
            dataset.append({'question': user_input, 'answer': edited_response})
           
            save_knowledge_base(knowledge_base_path, {'questions': dataset})
            print('Thank you for the correction. The knowledge base has been updated.')

        print(f'Bot: {answer}')

if __name__ == '__main__':
    qa_dataset = load_qa_dataset('/home/josse/informatika/proyek python/databases/irrelevant datas/Conversation.csv')
    knowledge_base_path = "knowledge_base.json"
    chat_bot(qa_dataset, knowledge_base_path)