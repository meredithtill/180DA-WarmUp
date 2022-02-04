#from https://pypi.org/project/SpeechRecognition/
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    audio_data = r.record(source, duration=10)
    print("Recognizing...")
    text = r.recognize_google(audio_data)
    print(text)
