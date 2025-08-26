# transcriber.py
import os
import tempfile
import subprocess
from typing import Union, Dict, Any
from pydub import AudioSegment  # requires ffmpeg

def _convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to mono 16k WAV using pydub (ffmpeg).
    Returns path to temporary wav file.
    """
    out_path = tempfile.mktemp(suffix=".wav")
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(out_path, format="wav")
        return out_path
    except Exception as e_pydub:
        # fallback to ffmpeg CLI
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_path
        except Exception as e_ffmpeg:
            if os.path.exists(out_path):
                try: os.remove(out_path)
                except: pass
            raise RuntimeError(f"Conversion failed. pydub: {e_pydub}; ffmpeg: {e_ffmpeg}")

def transcribe_audio(source: Union[str, "UploadFile"], model) -> Dict[str, Any]:
    """
    Accepts a path (str) or a Streamlit UploadFile (file-like) and a pre-loaded Whisper model.
    Returns dict: {"ok":bool,"error":str|None,"text":str,"segments":list}
    """
    tmp_input = None
    remove_input = False

    # If user passed an UploadFile object, write to temp file
    if not isinstance(source, str) and hasattr(source, "read"):
        suffix = "." + source.name.split(".")[-1] if "." in source.name else ".tmp"
        tmp_input = tempfile.mktemp(suffix=suffix)
        with open(tmp_input, "wb") as f:
            f.write(source.getbuffer() if hasattr(source, "getbuffer") else source.read())
        input_path = tmp_input
        remove_input = True
    else:
        input_path = source

    if not os.path.exists(input_path):
        return {"ok": False, "error": f"File not found: {input_path}", "text": "", "segments": []}

    # Convert to wav (16k mono)
    needs_cleanup_wav = False
    wav_path = input_path
    try:
        if not input_path.lower().endswith(".wav"):
            wav_path = _convert_to_wav(input_path)
            needs_cleanup_wav = True
    except Exception as e:
        if remove_input and tmp_input and os.path.exists(tmp_input):
            try: os.remove(tmp_input)
            except: pass
        return {"ok": False, "error": f"Conversion to WAV failed: {e}", "text": "", "segments": []}

    # Run transcription (Whisper)
    try:
        result = model.transcribe(wav_path)
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        out = {"ok": True, "error": None, "text": text, "segments": segments}
    except Exception as e:
        out = {"ok": False, "error": f"Transcription failed: {e}", "text": "", "segments": []}

    # cleanup temporaries
    if needs_cleanup_wav and os.path.exists(wav_path):
        try: os.remove(wav_path)
        except: pass
    if remove_input and tmp_input and os.path.exists(tmp_input):
        try: os.remove(tmp_input)
        except: pass

    return out
