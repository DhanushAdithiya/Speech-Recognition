import asyncio
from re import A, I
from dotenv import load_dotenv
import discord
import os

load_dotenv()
token = os.getenv("TOKEN")
user_id = os.getenv("USER_ID")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents = intents)

@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

async def ping_user(message):
        await client.wait_until_ready()
        channel = client.get_channel(1099337838237061185)
        print(channel)
        await channel.send(f"<@" + str(user_id) + "> There is a " + message)
        await asyncio.sleep()

client.run(token)
