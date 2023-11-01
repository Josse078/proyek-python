from transformers import GPT2Tokenizer, GPT2LMHeadModel


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    response = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(response[0], skip_special_tokens=True)

def chat_bot():
    print("Bot: Hai! Tanya saya apa saja (quit untuk keluar)")
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break
        response = generate_response(user_input)
        print(f'Bot: {response}')

if __name__ == '__main__':
    chat_bot()