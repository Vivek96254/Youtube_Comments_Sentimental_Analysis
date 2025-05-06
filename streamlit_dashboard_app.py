import re
import os
import nltk
import torch
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from googleapiclient.discovery import build

nltk.download("stopwords")
nltk.download("wordnet")

# Load RoBERTa model and tokenizer from pickle
@st.cache_resource
def load_roberta_from_pickle(pickle_path="roberta_model.pkl"):
    with open(pickle_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('&quot;', '"').replace('&amp;', '&')
    text = re.sub(r'&#39;', "'", text)
    text = re.sub(r'\b(?:href|rest api|json|api|http|https|www|endpoint|variable|function|code|error'
                  '|response|request|post|get|put|delete|cli|command line|url|key|value|object|array|parameter)\b', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,:;\'"()-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

# Fetch YouTube Comments
def get_comments(video_id, max_total=100):
    API_KEY = st.secrets["YOUTUBE_API_KEY"] if "YOUTUBE_API_KEY" in st.secrets else os.getenv("YOUTUBE_API_KEY")
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    comments = []
    next_page_token = None
    while len(comments) < max_total:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_total - len(comments)),
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_total:
                break
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return comments

# VADER Analysis
def vader_analysis(cleaned_comments, original_comments):
    sid = SentimentIntensityAnalyzer()
    vader_results = []
    for i, cleaned in enumerate(cleaned_comments):
        if cleaned.strip():
            sentiment = sid.polarity_scores(cleaned)
            compound = sentiment['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            vader_results.append((original_comments[i].strip(), compound, label))
    return vader_results

# RoBERTa Analysis using pickle-loaded model
def roberta_analysis(comments, tokenizer, model, device):
    results = []
    for comment in comments:
        inputs = tokenizer(comment, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        sentiment = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        label = ['negative', 'neutral', 'positive'][sentiment.argmax()]
        results.append((comment, sentiment.max(), label))
    return results

# Streamlit App
st.title("YouTube Comments Sentiment Analysis Dashboard")

video_url = st.text_input("Enter YouTube Video URL")
max_comments = st.slider("Number of comments to fetch", 10, 500, 100)

if st.button("Analyze Comments") and video_url:
    tokenizer, model, device = load_roberta_from_pickle()
    video_id = video_url[-11:]
    original_comments = get_comments(video_id, max_comments)
    cleaned_comments = [preprocess_text(comment) for comment in original_comments]

    st.success(f"Fetched and cleaned {len(cleaned_comments)} comments")

    # WordCloud
    all_words = " ".join(cleaned_comments).split()
    word_freq = Counter(all_words)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_words))
    st.subheader("Word Cloud of Comments")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # VADER
    vader_results = vader_analysis(cleaned_comments, original_comments)
    vader_counts = Counter([label for _, _, label in vader_results])
    st.subheader("VADER Sentiment Distribution")
    st.bar_chart(vader_counts)

    # RoBERTa
    roberta_results = roberta_analysis(original_comments, tokenizer, model, device)
    roberta_counts = Counter([label for _, _, label in roberta_results])
    st.subheader("RoBERTa Sentiment Distribution")
    st.bar_chart(roberta_counts)

    # Show Top Comments
    st.subheader("Top 5 Positive and Negative Comments")
    top_positive = [c for c, s, l in roberta_results if l == 'positive'][:5]
    top_negative = [c for c, s, l in roberta_results if l == 'negative'][:5]

    st.markdown("**Top Positive Comments:**")
    for comment in top_positive:
        st.write(comment)

    st.markdown("**Top Negative Comments:**")
    for comment in top_negative:
        st.write(comment)

    st.success("Analysis complete!")
