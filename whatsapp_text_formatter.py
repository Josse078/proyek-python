# #remove timestamps : step 1,setelah itu pake conversation pairing untuk pair hasilnya
# import re


# input_file = "/home/josse/informatika/separated_chat.txt"
# output_file = "cleaned_chat.txt"


# with open(input_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
#     for line in input_file:
#         line = line.strip()

        
#         line = re.sub(r'\[\d{2}/\d{2}/\d{2} \d{2}.\d{2}.\d{2}\] .*: ', '', line)

       
#         output_file.write(line + '\n')

# print("Timestamps removed. Cleaned chat saved as", output_file)

# input_file = "/home/josse/informatika/cleaned_chat.txt"  
# output_file = "conversation_pairs.txt"


# with open(input_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
#     lines = input_file.readlines()
#     i = 0
#     while i < len(lines):
#         question = lines[i].strip()
#         i += 1
#         if i < len(lines):
#             answer = lines[i].strip()
#             i += 1
#             output_file.write(f"{question},{answer}\n")

# print("Conversation pairs created. Saved as", output_file)
# #menghasilkan pasangan dari cleaned_txt:step 2, lalu remove image atau missed call menggunakan misc_remover.py


# input_file = '/home/josse/informatika/conversation_pairs.txt'  
# output_file = "chat_without_misc.txt"


# with open(input_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
#     for line in input_file:
#         line = line.strip()
        
        
#         unwanted_phrases = ["Missed voice call", "image omitted", "sticker omitted"]
#         if not any(phrase in line for phrase in unwanted_phrases):
#             output_file.write(line + '\n')

# print("Chat without unwanted messages saved as", output_file)
# #menghilangkan objek misc seperti calls atau images:step 3, dan jangan lupa mengganti data menjadi csv dan diatasnya tulis question,answer


import re
import csv

input_file = "/home/josse/informatika/proyek python/databases/_chat.txt"
output_file = "cleaned_chat.csv"


current_speaker = None
current_message = ""
conversations = []

with open(input_file, "r", encoding="utf-8") as input_file:
    for line in input_file:
        line = line.strip()
        
     
        match = re.match(r'\[\d{2}/\d{2}/\d{2} \d{2}.\d{2}.\d{2}\] (.*): (.*)', line)
        if match:
            speaker, message = match.groups()
            
            
            if current_speaker and current_speaker != speaker:
                conversations.append([current_message, message])
                current_message = ""
            
            
            current_speaker = speaker
            if current_message:
                current_message += " " + message  
            else:
                current_message = message
    
  
    if current_speaker:
        conversations.append([current_message, ""])


with open(output_file, "w", encoding="utf-8", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["question", "answer"])
    for conversation in conversations:
        csv_writer.writerow([conversation[0], conversation[1]])

print("Cleaned chat converted to CSV format. CSV file saved as", output_file)