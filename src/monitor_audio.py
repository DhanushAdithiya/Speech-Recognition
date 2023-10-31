
import pyaudio
import numpy as np
import wave
import audioop

threshold = 50
n_recordings = 0


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, rate=44100, channels=1, input=True, output=False)

recording = False
recorded_audio = []

while True:
    data = stream.read(1024)
    audio_array = np.frombuffer(data, dtype=np.int16)

    rms = audioop.rms(data, 2)
    # print(rms)  
    if rms > threshold:
        if not recording:
            recorded_audio = audio_array.copy()
            recording = True
        else:
            recorded_audio = np.append(recorded_audio, audio_array)

    elif recording:
        recording = False

        if len(recorded_audio) > 22050:  # Check for at least 1 second of recorded audio
            wave_file = wave.open(f"../temp_audios/temp_audio{n_recordings}.wav", "wb")
            wave_file.setnchannels(1)
            wave_file.setsampwidth(2)
            wave_file.setframerate(44100)
            wave_file.writeframes(recorded_audio.tobytes())
            wave_file.close()
            n_recordings += 1
            print("Audio recorded")

            

stream.stop_stream()
stream.close()