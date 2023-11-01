
import os
import json
from difflib import get_close_matches
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy  
from sentence_transformers import SentenceTransformer, util  


tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = BertForQuestionAnswering.from_pretrained("indobenchmark/indobert-base-p1")
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return " ".join(tokens)

def load_knowledge_base(file_path:str) -> dict:
    with open(file_path,'r') as file:
        data : dict = json.load(file)
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
def save_knowledge_base(file_path:str,data:dict):
    with open(file_path,'w') as file: 
        json.dump(data,file,indent=2)
def find_best_match(user_question:str,questions:list[str]) -> str | None:
    matches:list = get_close_matches(user_question,questions,n=1,cutoff = 0.6)
    return matches[0] if matches else None
def get_answer_for(question: str,knowledge_base:dict) -> str | None:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]


def answer_question(question, dataset):
    similar_question = find_similar_question(question, dataset)
    user_input = preprocess_text(question)  
    user_input_embedding = sentence_model.encode(user_input)

    question_embeddings = [sentence_model.encode(preprocess_text(entry['question'])) for entry in dataset]
    similarities = util.pytorch_cos_sim(user_input_embedding, question_embeddings)

    most_similar_index = similarities.argmax()
    most_similar_question = dataset[most_similar_index]['question']
    most_similar_answer = dataset[most_similar_index]['answer']

    if similarities[most_similar_index] > 0.6:  
        return most_similar_answer
    for entry in dataset:
        if similar_question.lower() == entry['question'].lower() or similar_question.lower() == entry['answer'].lower():
            return entry['answer']

    inputs = tokenizer(question, similar_question, return_tensors="pt", padding=True, truncation=True)
    start_scores, end_scores = model(**inputs)
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index + 1]))

    return answer if answer else "I'm not sure how to answer that."
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
qa_dataset = load_qa_dataset('/home/josse/informatika/proyek python/databases/merged_file.csv')
#qa_dataset = load_qa_dataset('/home/josse/informatika/proyek python/databases/irrelevant datas/Conversation.csv')
def chat_bot(dataset):
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break
        answer = answer_question(user_input, dataset)
        print(f'Bot: {answer}')
if __name__ == '__main__':
    chat_bot(qa_dataset)
