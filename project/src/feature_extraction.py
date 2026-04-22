import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import tldextract
import math

def calculate_entropy(text):
    if not text:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(text.count(chr(x))) / len(text)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_features_from_url(url):
    """
    Extracts advanced lexical features from a given URL string.
    Returns a dictionary of features.
    """
    # Normalize for counting
    clean_url = re.sub(r'^https?://', '', url)
    
    parsed_url = urlparse(url)
    ext = tldextract.extract(url)
    
    # 1. Length Features
    url_len = len(url)
    hostname_len = len(parsed_url.netloc) if parsed_url.netloc else len(clean_url.split('/')[0])
    path_len = len(parsed_url.path) if parsed_url.path else 0
    
    # 2. Count Features
    count_dot = clean_url.count('.')
    count_hyphen = clean_url.count('-')
    count_at = clean_url.count('@')
    count_question = clean_url.count('?')
    count_equals = clean_url.count('=')
    count_ampersand = clean_url.count('&')
    count_digits = sum(c.isdigit() for c in clean_url)
    count_letters = sum(c.isalpha() for c in clean_url)
    
    # 3. Binary Indicators
    use_of_ip = 1 if re.search(
        r'\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', url
    ) else 0
    
    is_https = 1 if url.startswith('https') else 0
    
    shorteners = r'\b(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|' \
                 r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|' \
                 r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|' \
                 r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.me)\b'
    use_shortener = 1 if re.search(shorteners, url.lower()) else 0

    # 4. Keyword and Semantic Features
    suspicious_words = ['login', 'signin', 'bank', 'account', 'update', 'free', 'lucky', 'bonus', 'verification', 'verify', 'secure']
    count_suspicious = sum(1 for word in suspicious_words if word in url.lower())
    
    # 5. Domain and Structure
    dir_depth = clean_url.count('/')
    num_subdomains = len(ext.subdomain.split('.')) if ext.subdomain else 0
    tld_len = len(ext.suffix) if ext.suffix else 0
    
    # 6. Statistical
    url_entropy = calculate_entropy(url)
    digit_ratio = count_digits / url_len if url_len > 0 else 0
    
    return {
        'url_len': url_len,
        'hostname_len': hostname_len,
        'path_len': path_len,
        'count_dot': count_dot,
        'count_hyphen': count_hyphen,
        'count_at': count_at,
        'count_question': count_question,
        'count_equals': count_equals,
        'count_ampersand': count_ampersand,
        'count_digits': count_digits,
        'count_letters': count_letters,
        'use_of_ip': use_of_ip,
        'is_https': is_https,
        'use_shortener': use_shortener,
        'count_suspicious': count_suspicious,
        'dir_depth': dir_depth,
        'num_subdomains': num_subdomains,
        'tld_len': tld_len,
        'url_entropy': url_entropy,
        'digit_ratio': digit_ratio
    }

def apply_feature_extraction(df):
    """
    Applies the extraction function across the entire pandas DataFrame.
    """
    features_list = df['url'].apply(extract_features_from_url)
    features_df = pd.DataFrame(features_list.tolist())
    
    # Concatenate the features with original labels
    result_df = pd.concat([features_df, df['label']], axis=1)
    
    # Drop rows with na if any generated during extraction
    result_df = result_df.dropna()
    
    # Separate X and y
    X = result_df.drop('label', axis=1)
    y = result_df['label']
    
    return X, y
