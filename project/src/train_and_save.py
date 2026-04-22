import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_and_preprocess_data
from src.feature_extraction import apply_feature_extraction
from src.models import get_random_forest, train_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_best_model():
    dataset_path = 'malicious_phish.csv'
    model_dir = 'models'
    MAX_ROWS = None  # Training on the FULL dataset

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. Load and Preprocess
    logging.info("Loading and preprocessing data...")
    df = load_and_preprocess_data(dataset_path, max_rows=MAX_ROWS)
    if df is None:
        return

    # 2. Feature Extraction
    logging.info("Extracting features...")
    X, y = apply_feature_extraction(df)

    # 3. Train-Test Split (just for validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    logging.info("Training Random Forest model...")
    model = get_random_forest()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    logging.info(f"Model validation accuracy: {score:.4f}")

    # 5. Save Model
    model_path = os.path.join(model_dir, 'best_model.joblib')
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_best_model()
