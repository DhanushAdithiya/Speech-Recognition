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

from HiddenMarkovModelTrainer import HiddenMarkovModelTrainer
from generateModel import generate_model
from prediction import predict


if not os.listdir("../model"):
    hmm_models = generate_model("../Training2")
else:
    hmm_models = []
    for file in os.listdir("../model"):
        filename = file[:-4]
        model = pickle.load(open(f"../model/{file}", "rb"))
        hmm_models.append((model, filename))

load_dotenv()
token = os.getenv("TOKEN")
user_id = os.getenv("USER_ID")


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents = intents)

# @client.event  
# async def on_ready():
#     while True:
#         pred = predict("../test/knock.mp3", hmm_models)
#         await client.get_channel(1099337838237061185).send("You have a " + pred )
#         await asyncio.sleep(3600)


# client.run(token)



# Options
# 1 - Record every 2 seconds and analyze that - DRAWBACK - WILL LEAD TO MORE OVERHEAD + POSSIBLE DATA LOSS + SLOW
# 2 - Stream the audio continously and have the model monitor it continiously - IDK HOW?

# class Application():
#     def __init__(self) -> None:
#         self.p = pyaudio.PyAudio()
#         self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True)
    
#     async def on_ready(self):
#         self.task = asyncio.create_task(self.listen())

#     async def listen(self):
#         while True:
#             data = self.stream.read(1024)
            
#             if predict()


