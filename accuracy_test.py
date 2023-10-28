import csv
import torch
from transformers import BertForSequenceClassification, BertTokenizer


question_answer_pairs = []

with open('/home/josse/informatika/proyek python/databases/ezra_louis_merged_file.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        question = row['question']
        answer = row['answer']
        question_answer_pairs.append((question, answer))


model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


accurate_responses = []

for question, answer in question_answer_pairs:
  
    inputs = tokenizer(question, answer, return_tensors="pt", padding=True, truncation=True, max_length=512)

   
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

    if predicted_label == 1:
        accurate_responses.append((question, answer))


with open('accurate_responses.csv', 'w', newline='') as csv_file:
    fieldnames = ['question', 'answer']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for question, answer in accurate_responses:
        writer.writerow({'question': question, 'answer': answer})
print('done')