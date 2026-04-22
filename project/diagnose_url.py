import tldextract
import joblib
import pandas as pd
import os
from src.feature_extraction import extract_features_from_url

TRUSTED_BRANDS = {
    "google", "chatgpt", "openai", "facebook", "apple", "microsoft", "amazon", 
    "youtube", "instagram", "twitter", "linkedin", "netflix", "github", 
    "stackoverflow", "wikipedia", "yahoo", "bing", "reddit", "paypal", "chase", 
    "wellsfargo", "bankofamerica", "icloud", "ycombinator", "vercel", "netlify",
    "medium", "nytimes", "cnn", "bbc", "forbes", "bloomberg", "reuters", 
    "techcrunch", "wired", "theverge", "quora", "twitch", "discord", "spotify", 
    "adobe", "dropbox"
}

def diagnose(url):
    print(f"Diagnosing: {url}")
    ext = tldextract.extract(url)
    print(f"Domain extracted: '{ext.domain}'")
    print(f"Is in TRUSTED_BRANDS: {ext.domain in TRUSTED_BRANDS}")
    
    features = extract_features_from_url(url)
    print("Features extracted:")
    for k, v in features.items():
        print(f"  {k}: {v}")
        
    MODEL_PATH = "models/best_model.joblib"
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        features_df = pd.DataFrame([features])
        expected_columns = model.feature_names_in_
        features_df = features_df[expected_columns]
        prob = model.predict_proba(features_df)[0][1]
        print(f"Model Raw Probability: {prob}")
    else:
        print("Model not found!")

if __name__ == "__main__":
    diagnose("https://chatgpt.com")
    print("-" * 20)
    diagnose("chatgpt.com")
