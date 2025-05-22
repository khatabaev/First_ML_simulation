from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

audio_exemp = '/Users/khatabaev/Desktop/First_try.wav'


result = pipe(audio_exemp)


print(result)