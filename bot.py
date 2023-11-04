import discord
from main import answer_question,qa_dataset
intents = discord.Intents.default()
intents.typing = False
intents.presences = False

async def send_message(message,user_message,is_private):
    try:
        answer = answer_question(user_message,qa_dataset)
        await message.author.send(answer) if is_private else await message.channel.send(answer)
    except Exception as e:
        print(e)
def run_discord_bot():
    token = "MTE3MDM1MTI2OTYzOTA5ODQyOQ.GQAUyk.HN0HVaw9PHhMqRQnlcglh0IU4MVcHO-cKwPxRA"
    client = discord.Client(intents = intents)

    @client.event
    async def on_ready():
        print(f'{client.user} is now running')
    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        username = str(message.author)
        user_message = str(message.content)
        channel = str(message.channel)
        print(f"{username}: said '{user_message}' {channel}")
        if user_message[0] == '?':
            user_message = user_message[1:]
            await send_message(message,user_message,is_private = True)
        else:
            await send_message(message,user_message,is_private = False)
    client.run(token)


