import os
import numpy as np
import sounddevice as sd
from openai import OpenAI
from io import BytesIO
import soundfile as sf

# OpenAI Setup
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Conversation Memory
conversation = [
    {"role": "system", "content": "You are a AI voice assistant. Keep replies concise and clear."}
]

memory_texts = []

def store_memory(user_text):
    memory_texts.append(user_text)

def retrieve_memory(user_text, max_items=5):
    """Return last few messages that might be relevant to current input."""
    return memory_texts[-max_items:]

# Audio Recording
def record_audio(duration=4, samplerate=16000):
    print("Listening... Speak now!", flush=True)
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.", flush=True)
    return audio.flatten()

# Speech-to-Text
def speech_to_text(audio):
    buffer = BytesIO()
    sf.write(buffer, audio, 16000, format='WAV')
    buffer.seek(0)

    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=buffer,
            language="en"
        )
        text = transcript.text.strip()
        print("You:", text, flush=True)
        return text
    except Exception as e:
        print("Error in transcription:", e, flush=True)
        return ""

# GPT Interaction
def ask_gpt(user_text):
    context_memory = "\n".join(retrieve_memory(user_text))
    conversation.append({"role": "user", "content": f"{user_text}\nContext:\n{context_memory}"})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation
        )
        reply = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": reply})
        store_memory(user_text)
        print("Assistant:", reply, flush=True)
        return reply
    except Exception as e:
        print("Error in GPT response:", e, flush=True)
        return "Sorry, I couldn't process that."

# Text-to-Speech
def speak(text):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        audio_data = np.frombuffer(response.content, dtype=np.int16)
        sd.play(audio_data, samplerate=24000)
        sd.wait()
    except Exception as e:
        print("Error in TTS:", e, flush=True)

# Main Loop
def run():
    print("AI Voice Agent running! Say 'stop' to exit.", flush=True)
    while True:
        audio = record_audio()
        user_text = speech_to_text(audio)

        if user_text == "":
            continue
        if "stop" in user_text.lower():
            print("Exiting AI Voice Agent. Goodbye!", flush=True)
            break

        reply = ask_gpt(user_text)
        speak(reply)

if __name__ == "__main__":
    run()