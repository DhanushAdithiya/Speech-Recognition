from platform import java_ver
import numpy as np
from scipy.io import wavfile
import os
import pyaudio
import pickle
import wave
import sys
import asyncio
from dotenv import load_dotenv
import discord
import time
import threading
import signal

from HiddenMarkovModelTrainer import HiddenMarkovModelTrainer
from generateModel import generate_model
from prediction import predict
from monitor_audio import monitor_audio

# Load environment variables from a .env file
load_dotenv()

# Your Discord bot token
TOKEN = os.getenv('TOKEN')

# Discord client
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

hmm_models = []


if not os.listdir("../model"):
        hmm_models = generate_model("../Training2")
else:
    for file in os.listdir("../model"):
        filename = file[:-4]
        model = pickle.load(open(f"../model/{file}", "rb"))
        hmm_models.append((model, filename))


def watch(path_to_watch, event, text_channel):
    print("Starting to watch")
    while True:
        if not os.listdir(path_to_watch):
            time.sleep(1)
        for filename in os.listdir(path_to_watch):
            prediction = predict(f"{path_to_watch}/{filename}", hmm_models)
            os.remove(f"{path_to_watch}/{filename}")
            # Send the prediction as a message to the specified text channel
            asyncio.run_coroutine_threadsafe(text_channel.send(f"Prediction: {prediction}"), client.loop)
        if event.is_set():
            break

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    # Find the text channel where you want to send predictions
    text_channel = client.get_channel(int(os.getenv('CHANNEL_ID')))
    event1 = threading.Event()
    event2 = threading.Event()
    t1 = threading.Thread(target=watch, args=("../temp_audios", event1, text_channel,))
    t2 = threading.Thread(target=monitor_audio, args=(event2,) )
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()

# Start the Discord bot
client.run(TOKEN)