import re
import csv

input_file = "/home/josse/informatika/proyek python/databases/irrelevant datas/ezra_chat.txt"
output_file = "cleaned_chat.csv"

conversations = []
current_speaker = None
current_message = []

with open(input_file, "r", encoding="utf-8") as input_file:
    for line in input_file:
        line = line.strip()
        
        match = re.match(r'\[\d{2}/\d{2}/\d{2} \d{2}.\d{2}.\d{2}\] (.*): (.*)', line)
        if match:
            speaker, message = match.groups()
            
            if current_speaker and current_speaker != speaker:
                conversations.append([current_message])
                current_message = []
            
            current_speaker = speaker
            current_message.append(message)
    
    if current_speaker:
        conversations.append([current_message])

with open(output_file, "w", encoding="utf-8", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["question", "answer"])
    for i in range(0, len(conversations), 2):
        question = ' '.join(conversations[i][0])
        answer = ' '.join(conversations[i + 1][0]) if i + 1 < len(conversations) else ""
        csv_writer.writerow([question, answer])

print("Cleaned chat converted to CSV format. CSV file saved as", output_file)