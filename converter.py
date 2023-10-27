
import csv
import json


data = []


with open('/home/josse/informatika/proyek python/Conversation.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append({"question": row["question"], "answer": row["answer"]})


json_data = {"questions": data}


json_string = json.dumps(json_data, indent=2)


with open('output.json', 'w') as json_file:
    json_file.write(json_string)

print("Data converted and saved to output.json.")
