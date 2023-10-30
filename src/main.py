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

@client.event
async def on_ready():
    while True:
        pred = predict("../test/knock.mp3", hmm_models)
        await client.get_channel(1099337838237061185).send("You have a " + pred )
        await asyncio.sleep(3600)


client.run(token)




# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1 if sys.platform == "darwin" else 2
# RATE = 44100
# RECORD_SECONDS = 3


# with wave.open("output.mp3", "wb") as wf:
#     p = pyaudio.PyAudio()
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)


#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

#     print("RECORDING")
#     for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
#         wf.writeframes(stream.read(CHUNK))
#     print("DONE")


#     stream.close()
#     p.terminate()
    

