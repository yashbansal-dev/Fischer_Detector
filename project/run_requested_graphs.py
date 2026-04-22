import os
import logging
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_and_preprocess_data
from src.feature_extraction import apply_feature_extraction
from src.models import (
    get_voting_classifier, get_stacking_classifier, get_kmeans_hybrid_pipeline, train_model
)
from src.evaluation import evaluate_model, plot_confusion_matrix

def main():
    dataset_path = 'malicious_phish.csv'
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    logging.info("Running specific enesembles for user graphs...")
    df = load_and_preprocess_data(dataset_path, binary_classification=True)
    X, y = apply_feature_extraction(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models_to_train = {
        "Voting Classifier": get_voting_classifier(),
        "Stacking (KNN+SVM -> RF)": get_stacking_classifier(),
        "KMeans Hybrid (KMeans+LR)": get_kmeans_hybrid_pipeline()
    }

    for name, model in models_to_train.items():
        logging.info(f"--- Training {name} ---")
        try:
            model = train_model(name, model, X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Sanitize name for filename
            safe_name = "".join([c if c.isalnum() else "_" for c in name])
            cm_path = os.path.join(results_dir, f"{safe_name}_cm.png")
            plot_confusion_matrix(y_test, y_pred, model_name=name, save_path=cm_path)
            logging.info(f"Saved: {cm_path}")
        except Exception as e:
            logging.error(f"Error for {name}: {e}")

if __name__ == "__main__":
    main()
