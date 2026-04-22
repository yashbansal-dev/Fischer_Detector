import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(filepath, max_rows=None, binary_classification=True):
    """
    Loads dataset, drops nulls, ensures unique rows, and maps labels.
    """
    logging.info(f"Loading dataset from {filepath}...")
    try:
        if max_rows is not None:
            df = pd.read_csv(filepath, nrows=max_rows)
        else:
            df = pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"File {filepath} not found.")
        return None

    logging.info(f"Original dataset shape: {df.shape}")

    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop missing values
    df = df.dropna()

    logging.info(f"Shape after dropping duplicates and nulls: {df.shape}")

    if 'url' not in df.columns or 'type' not in df.columns:
        raise ValueError("Dataset must contain 'url' and 'type' columns.")

    if binary_classification:
        logging.info("Mapping to binary classification (benign -> 0, malicious -> 1).")
        # Assuming 'benign' is the only safe class, others are malicious variations
        df['label'] = df['type'].apply(lambda x: 0 if x.strip().lower() == 'benign' else 1)
    else:
        # Multiclass label encoding if needed
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['type'])

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info("Data preprocessing completed successfully.")
    return df
