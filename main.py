import os
import json
from difflib import get_close_matches
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
def load_knowledge_base(file_path:str) -> dict:
    with open(file_path,'r') as file:
        data : dict = json.load(file)
    return data

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

# def chat_bot():
#     knowledge_base: dict = load_knowledge_base('/home/josse/informatika/proyek python/output.json')
#     while True:
#         user_input: str = input('You :')

#         if user_input.lower() == 'quit':
#             break
        
#         best_match:str | None = find_best_match(user_input,[q["question"]for q in knowledge_base["questions"]])

#         if best_match:
#             answer: str = get_answer_for(best_match,knowledge_base)
#             print(f'Bot:{answer}')
#         else:
#             print('Bot: I don\'t know the answer.Can you teach me?')
#             new_answer: str = input('Type the answer of "skip" to skip')
#             if new_answer.lower() != 'skip':
#                 knowledge_base["questions"].append({"question": user_input,"answer":new_answer})
#                 save_knowledge_base('knowledge_base.json',knowledge_base)
#                 print('Bot: Thank you! I learned a new response')
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    start_scores, end_scores = model(**inputs).values() 

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    
    start_index = int(start_index)
    end_index = int(end_index)

    input_ids = inputs["input_ids"][0].tolist()
    answer = tokenizer.decode(input_ids[start_index:end_index + 1])

    return answer

import json

def chat_bot():
    
    with open('/home/josse/informatika/proyek python/context.json', 'r') as file:
        context_data = json.load(file)
        context = context_data.get("context", "")

    while True:
        user_input = input('You: ')

        if user_input.lower() == 'quit':
            break

        answer = answer_question(user_input, context)
        print(f'Bot: {answer}')

if __name__ == '__main__':
    chat_bot()
