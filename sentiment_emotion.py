# sentiment_emotion.py
from typing import List, Dict, Any
from collections import Counter

# Lazy pipelines to speed cold-start
_sent_pipe = None
_emo_pipe = None

def _init_pipes():
    global _sent_pipe, _emo_pipe
    if _sent_pipe is None:
        # fast default sentiment pipeline; change model if you prefer
        from transformers import pipeline
        _sent_pipe = pipeline("sentiment-analysis")
    if _emo_pipe is None:
        from transformers import pipeline
        # j-hartmann/emotion-english-distilroberta-base is good; HF will download
        _emo_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    return _sent_pipe, _emo_pipe

def analyze_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Input: segments = [{"start":float,"end":float,"text":str,"speaker":...}, ...]
    Returns: {
      "per_segment": [ {start,end,speaker,text,sentiment,sent_score,emotion,emo_score}, ... ],
      "aggregate": { "sentiment_distribution": {...}, "emotion_distribution": {...}, "num_segments": N}
    }
    """
    sent_pipe, emo_pipe = _init_pipes()

    per_segment = []
    sent_counts = Counter()
    emo_counts = Counter()

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # sentiment
        try:
            s = sent_pipe(text)[0]  # e.g. {'label': 'POSITIVE', 'score': 0.99}
            sentiment = s.get("label")
            sent_score = float(s.get("score", 0.0))
        except Exception:
            sentiment = "UNKNOWN"
            sent_score = 0.0

        # emotion (returns list of labels with scores)
        try:
            emo_scores = emo_pipe(text)[0]  # list of dicts
            if isinstance(emo_scores, list) and len(emo_scores) > 0:
                top = max(emo_scores, key=lambda x: x["score"])
                emotion = top["label"]
                emo_score = float(top["score"])
            else:
                emotion = "neutral"
                emo_score = 0.0
        except Exception:
            emotion = "UNKNOWN"
            emo_score = 0.0

        per = {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "speaker": seg.get("speaker"),
            "text": text,
            "sentiment": sentiment,
            "sentiment_score": sent_score,
            "emotion": emotion,
            "emotion_score": emo_score
        }
        per_segment.append(per)
        sent_counts[sentiment] += 1
        emo_counts[emotion] += 1

    aggregate = {
        "sentiment_distribution": dict(sent_counts),
        "emotion_distribution": dict(emo_counts),
        "num_segments": len(per_segment)
    }

    return {"per_segment": per_segment, "aggregate": aggregate}
