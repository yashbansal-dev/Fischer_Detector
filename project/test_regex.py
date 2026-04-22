import re
import tldextract
import joblib
import pandas as pd
import os
from src.feature_extraction import extract_features_from_url

shorteners = r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|' \
             r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|' \
             r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|' \
             r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.me'

def diagnose(url):
    print(f"Diagnosing: {url}")
    match = re.search(shorteners, url)
    if match:
        print(f"Shortener matched: '{match.group(0)}'")
    else:
        print("No shortener matched.")
    
    features = extract_features_from_url(url)
    print(f"use_shortener feature: {features['use_shortener']}")

if __name__ == "__main__":
    diagnose("https://chatgpt.com")
