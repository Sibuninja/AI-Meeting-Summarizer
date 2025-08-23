import whisper
import soundfile as sf   # <-- this was missing!

model = whisper.load_model("small")

def transcribe_audio(audio_path: str) -> str:
    # Check if audio has valid frames
    try:
        data, samplerate = sf.read(audio_path)
        if len(data) == 0:
            return "❌ Error: Empty or unreadable audio file."
    except Exception as e:
        return f"❌ Error reading audio: {e}"

    # If okay, run Whisper
    result = model.transcribe(audio_path)
    return result["text"]
