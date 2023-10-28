#remove timestamps : step 1,setelah itu pake conversation pairing untuk pair hasilnya
import re


input_file = "/home/josse/informatika/proyek python/databases/ezra_chat.txt"
output_file = "cleaned_chat.txt"


with open(input_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
    for line in input_file:
        line = line.strip()

        
        line = re.sub(r'\[\d{2}/\d{2}/\d{2} \d{2}.\d{2}.\d{2}\] .*: ', '', line)

       
        output_file.write(line + '\n')

print("Timestamps removed. Cleaned chat saved as", output_file)

input_file = "/home/josse/informatika/cleaned_chat.txt"  
output_file = "conversation_pairs.txt"


with open(input_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
    lines = input_file.readlines()
    i = 0
    while i < len(lines):
        question = lines[i].strip()
        i += 1
        if i < len(lines):
            answer = lines[i].strip()
            i += 1
            output_file.write(f"{question},{answer}\n")

print("Conversation pairs created. Saved as", output_file)
#menghasilkan pasangan dari cleaned_txt:step 2, lalu remove image atau missed call menggunakan misc_remover.py


input_file = '/home/josse/informatika/conversation_pairs.txt'  
output_file = "chat_without_misc.txt"


with open(input_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
    for line in input_file:
        line = line.strip()
        
        
        unwanted_phrases = ["Missed voice call", "image omitted", "sticker omitted"]
        if not any(phrase in line for phrase in unwanted_phrases):
            output_file.write(line + '\n')

print("Chat without unwanted messages saved as", output_file)
#menghilangkan objek misc seperti calls atau images:step 3, dan jangan lupa mengganti data menjadi csv dan diatasnya tulis question,answer