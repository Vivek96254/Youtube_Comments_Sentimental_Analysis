import logging
from fastapi import FastAPI
from typing import Tuple
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from googleapiclient.discovery import build
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RoBERTa Model and Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./roberta_sentiment')
model = RobertaForSequenceClassification.from_pretrained('./roberta_sentiment')
model.eval()

# YouTube API Initialization
API_KEY = "AIzaSyC-0-EclG9LWneAH6EnBGKmCHo0MNiOKk4"
youtube = build('youtube', 'v3', developerKey=API_KEY)
logging.info("Starting sentiment analysis...")

# Sentiment Analysis Function
def analyze_sentiment(comment: str) -> Tuple[str, float]:
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    score = probabilities[0][predicted_class].item()
    label = "Positive" if predicted_class == 1 else "Negative"
    return label, score


# Get Comments from YouTube
def get_comments(video_id):
    comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    logging.info(f"Retrieved Comments: {comments}")
    return comments

@app.get("/test")
def test_route():
    logging.info("Test route accessed.")
    return {"message": "Hello, world!"}

@app.get("/analyze/{video_id}")
def analyze_video(video_id: str):
    try:
        comments = get_comments(video_id)
        logging.info(f"Analyzing Comments: {comments}")
        sentiment_results = []
        for comment in comments:
            label, score = analyze_sentiment(comment)
            sentiment_results.append((comment, label, score))
        sentiment_results.sort(key=lambda x: x[2], reverse=True)
        top_positive = [c[0] for c in sentiment_results if c[1] == "Positive"][:10]
        top_negative = [c[0] for c in sentiment_results if c[1] == "Negative"][:10]
        return {"positive": top_positive, "negative": top_negative}
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"error": str(e)}
