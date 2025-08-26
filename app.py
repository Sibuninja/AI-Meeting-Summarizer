# app.py
import os
import time
import shutil
import streamlit as st
import whisper
import torch

from transcriber import transcribe_audio
from summarizer import summarize_text

# ---------- Config ----------
UPLOAD_DIR = "uploads"
HISTORY_DIR = "history"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Delete old uploads on app start to free disk space
# (If you prefer, comment this out to keep uploads)
for f in os.listdir(UPLOAD_DIR):
    try:
        fp = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)
        else:
            shutil.rmtree(fp)
    except Exception:
        pass

st.set_page_config(page_title="AI Meeting Summarizer", layout="wide")
st.title("🎙️ AI Meeting Summarizer")

# Sidebar: device + options
st.sidebar.header("Settings")
device_choice = st.sidebar.selectbox("Processing device", ["Auto (best)", "CPU", "GPU (CUDA)"])
prefer_gpu = False
if device_choice == "GPU (CUDA)":
    prefer_gpu = True
elif device_choice == "Auto (best)":
    prefer_gpu = torch.cuda.is_available()
else:
    prefer_gpu = False

# show detected status
st.sidebar.write("Torch CUDA available:", torch.cuda.is_available())
st.sidebar.write("Using GPU?:", prefer_gpu)

# button to clear saved history (manual)
if st.sidebar.button("Clear saved history (delete history/)"):
    try:
        if os.path.exists(HISTORY_DIR):
            shutil.rmtree(HISTORY_DIR)
        os.makedirs(HISTORY_DIR, exist_ok=True)
        st.sidebar.success("History cleared.")
    except Exception as e:
        st.sidebar.error(f"Failed to clear history: {e}")

# File uploader accepts audio & video
uploaded_file = st.file_uploader(
    "Upload audio or video (mp3,wav,m4a,mp4,mov,avi,mkv)",
    type=["mp3", "wav", "m4a", "mp4", "mov", "avi", "mkv"]
)

# summarizer choice
summary_model_choice = st.sidebar.selectbox(
    "Summarizer model (Auto recommended)",
    ["Auto (recommended)", "Pegasus", "Flan-T5", "Longformer"]
)

# Load Whisper model once and cache it
@st.cache_resource
def load_whisper(device_gpu: bool):
    dev = "cuda" if device_gpu and torch.cuda.is_available() else "cpu"
    # Whisper 'device' param accepts "cpu" or "cuda"
    return whisper.load_model("base", device=dev)

whisper_model = load_whisper(prefer_gpu)

# Main flow
if uploaded_file is not None:
    # unique filename
    fname = uploaded_file.name
    ts = int(time.time())
    saved_path = os.path.join(UPLOAD_DIR, f"{ts}_{fname}")
    with open(saved_path, "wb") as out:
        out.write(uploaded_file.getbuffer())

    st.success(f"Saved upload: {saved_path}")
    st.info("Converting and transcribing... this can take a while for long files.")

    # transcribe (transcriber handles conversion)
    result = transcribe_audio(saved_path, whisper_model)
    if not result["ok"]:
        st.error(result["error"])
    else:
        transcript = result["text"]
        st.subheader("📜 Transcript")
        st.write(transcript)

        # Save transcript to history
        base = os.path.splitext(os.path.basename(saved_path))[0]
        tpath = os.path.join(HISTORY_DIR, f"{base}_transcript.txt")
        with open(tpath, "w", encoding="utf-8") as f:
            f.write(transcript)

        # summarization
        if st.button("Generate Precise Summary"):
            st.info("Summarizing... (multi-step abstractive)")
            summary = summarize_text(transcript, model_choice=summary_model_choice, prefer_gpu=prefer_gpu)
            st.subheader("📝 Summary — concise & precise")
            st.success(summary)

            # save summary
            spath = os.path.join(HISTORY_DIR, f"{base}_summary.txt")
            with open(spath, "w", encoding="utf-8") as f:
                f.write(summary)

            # offer downloads
            st.download_button("⬇️ Download Transcript", data=transcript, file_name=f"{base}_transcript.txt", mime="text/plain")
            st.download_button("⬇️ Download Summary", data=summary, file_name=f"{base}_summary.txt", mime="text/plain")

# show recent history
st.sidebar.header("Recent summaries")
history_files = sorted([f for f in os.listdir(HISTORY_DIR) if f.endswith("_summary.txt")])
if history_files:
    for hf in history_files[-5:][::-1]:
        try:
            with open(os.path.join(HISTORY_DIR, hf), "r", encoding="utf-8") as f:
                preview = f.read(800)
            st.sidebar.write(f"**{hf}**")
            st.sidebar.write(preview + ("..." if len(preview) >= 800 else ""))
        except Exception:
            pass
else:
    st.sidebar.write("No summaries yet.")
