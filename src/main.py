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


def main():
    hmm_models = []
    def watch(path_to_watch):
        print("Starting to watch")
        if not os.listdir(path_to_watch):
            time.sleep(1)
        for filename in os.listdir(path_to_watch):
            print("PRED:", predict(f"{path_to_watch}/{filename}", hmm_models))
            os.remove(f"{path_to_watch}/{filename}")

    
    if not os.listdir("../model"):
        hmm_models = generate_model("../Training2")
    else:
        for file in os.listdir("../model"):
            filename = file[:-4]
            model = pickle.load(open(f"../model/{file}", "rb"))
            hmm_models.append((model, filename))


    event = threading.Event()
    t1 = threading.Thread(target=watch, args=("../temp_audios",))
    t2 = threading.Thread(target=monitor_audio, args=(event,))

    t2.daemon = True
    t1.daemon = True

    t1.start()
    t2.start()

    try:
        while 1: 
            time.sleep(.1)
    except KeyboardInterrupt:
        print("Attempting to close threads")
        t1.join()
        event.set()
        t2.join()
        print("Closed Threads Sucessfully")

if __name__ == "__main__":
    main()


# load_dotenv()
# token = os.getenv("TOKEN")
# user_id = os.getenv("USER_ID")


# intents = discord.Intents.default()
# intents.message_content = True

# client = discord.Client(intents = intents)


# @client.event  
# async def on_ready():
#     while True:
#         prediction = predict("../test/db.mp3", hmm_models)
#         print(prediction)
#         match prediction:
#             case "Doorbell":
#                 await client.get_channel(1099337838237061185).send("You have a visitor")
#             case "Cooker":
#                 await client.get_channel(1099337838237061185).send("The cooker has blown a whistle")
#             case _: 
#                 pass
#         await asyncio.sleep(10)


# client.run(token)



