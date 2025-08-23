from transformers import pipeline

# Load summarizer (distilbart is lightweight and good)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text: str, max_length=120, min_length=30) -> str:
    """
    Summarizes the input text.
    """
    summary = summarizer(
        text, 
        max_length=max_length, 
        min_length=min_length, 
        do_sample=False
    )
    return summary[0]['summary_text']
