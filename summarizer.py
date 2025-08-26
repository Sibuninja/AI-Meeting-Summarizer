# summarizer.py
from transformers import pipeline, AutoTokenizer
import torch
import math

# Choose a good abstractive model; Flan-T5-Base is a nice balance.
# You can change these ids if you prefer other models.
MODEL_IDS = {
    "Pegasus": "google/pegasus-cnn_dailymail",
    "Flan-T5": "google/flan-t5-base",
    "Longformer": "allenai/led-base-16384",
    "Auto": "google/flan-t5-base"
}

# cache pipelines per model id
_PIPELINES = {}

def _get_summarizer(model_name: str, device: int):
    """Return a pipeline instance cached per model_name."""
    key = f"{model_name}:{device}"
    if key in _PIPELINES:
        return _PIPELINES[key]
    # device: 0 for GPU, -1 for CPU
    summarizer = pipeline("summarization", model=model_name, device=device)
    _PIPELINES[key] = summarizer
    return summarizer

def _chunks_by_words(text: str, words_per_chunk: int = 600):
    words = text.split()
    for i in range(0, len(words), words_per_chunk):
        yield " ".join(words[i:i+words_per_chunk])

def summarize_text(text: str, model_choice: str = "Auto (recommended)", prefer_gpu: bool = False) -> str:
    """
    Multi-step abstractive summarization:
     - chunk transcript
     - summarize each chunk
     - combine chunk summaries and final-pass summarize
    """
    text = (text or "").strip()
    if not text:
        return "⚠️ No transcript to summarize."

    # determine device argument for HF pipeline
    device = 0 if (prefer_gpu and torch.cuda.is_available()) else -1

    # resolve model id
    if model_choice == "Pegasus":
        model_id = MODEL_IDS["Pegasus"]
    elif model_choice == "Flan-T5":
        model_id = MODEL_IDS["Flan-T5"]
    elif model_choice == "Longformer":
        model_id = MODEL_IDS["Longformer"]
    else:
        model_id = MODEL_IDS["Auto"]

    summarizer = _get_summarizer(model_id, device)

    # short text -> single pass
    word_count = len(text.split())
    if word_count < 500:
        out = summarizer(text, max_length=min(150, int(word_count*0.8)), min_length=30, do_sample=False)
        return out[0]["summary_text"]

    # chunked summarization
    chunk_summaries = []
    # choose chunk size depending on total length
    words_per_chunk = 700 if word_count < 3000 else 1200
    for chunk in _chunks_by_words(text, words_per_chunk):
        s = summarizer(chunk, max_length=200, min_length=40, do_sample=False)
        chunk_summaries.append(s[0]["summary_text"])

    # final pass: summarize the concatenated chunk summaries
    joined = " ".join(chunk_summaries)
    final = summarizer(joined, max_length=220, min_length=60, do_sample=False)
    return final[0]["summary_text"]
