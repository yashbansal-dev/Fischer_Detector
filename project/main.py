import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import from our custom modules
from src.data_preprocessing import load_and_preprocess_data
from src.feature_extraction import apply_feature_extraction, extract_features_from_url
from src.models import (
    get_logistic_regression, get_knn, get_svm, get_random_forest, 
    get_voting_classifier, get_stacking_classifier, get_kmeans_hybrid_pipeline, train_model
)
from src.evaluation import (
    evaluate_model, create_comparison_table, plot_confusion_matrix, 
    plot_model_comparison, plot_feature_importance, plot_precision_recall_comparison
)

def main():
    # Setup
    dataset_path = 'malicious_phish.csv'
    results_dir = 'results'
    # Use 50,000 for faster but comprehensive graph updates
    MAX_ROWS = 50000 
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1 & 2. Data Loading and Preprocessing
    logging.info(f"Step 1 & 2: Data Loading (Sample: {MAX_ROWS})")
    df = load_and_preprocess_data(dataset_path, max_rows=MAX_ROWS, binary_classification=True)
    if df is None: return

    # 3. Feature Extraction
    logging.info("Step 3: Feature Extraction")
    X, y = apply_feature_extraction(df)
    
    # 4. Train-Test Split
    logging.info("Step 4: Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # New: Visualize Data Distribution (Entropy)
    logging.info("Generating Data Distribution plots...")
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=X[y==0], x="url_entropy", label="Benign", fill=True)
    sns.kdeplot(data=X[y==1], x="url_entropy", label="Malicious", fill=True)
    plt.title("Distribution of URL Entropy (Benign vs Malicious)")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "entropy_distribution.png"))
    plt.close()

    # 5. Initialize Models
    models_to_train = {
        "Logistic Regression": get_logistic_regression(),
        "KNN": get_knn(),
        "Random Forest": get_random_forest(),
        "SVM": get_svm(),
        "Voting Classifier": get_voting_classifier(),
        "Stacking": get_stacking_classifier(),
        "KMeans Hybrid": get_kmeans_hybrid_pipeline()
    }

    results = []
    trained_models = {}

    # 5 & 6. Train and Evaluate
    for name, model in models_to_train.items():
        logging.info(f"--- Processing {name} ---")
        try:
            model = train_model(name, model, X_train, y_train)
            trained_models[name] = model
            y_pred = model.predict(X_test)
            
            eval_metrics = evaluate_model(y_test, y_pred, model_name=name)
            results.append(eval_metrics)
            
            safe_name = "".join([c if c.isalnum() else "_" for c in name])
            cm_path = os.path.join(results_dir, f"{safe_name}_cm.png")
            plot_confusion_matrix(y_test, y_pred, model_name=name, save_path=cm_path)
            
        except Exception as e:
            logging.error(f"Error processing {name}: {e}")

    # 7. Comparison Table & Plot
    logging.info("Generating Comparison Visuals...")
    comparison_df = create_comparison_table(results)
    comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    plot_model_comparison(comparison_df, os.path.join(results_dir, 'model_performance_comparison.png'))

    # New: Precision-Recall Curve Comparison
    plot_precision_recall_comparison(y_test, trained_models, X_test, os.path.join(results_dir, 'precision_recall_comparison.png'))

    # 8. Feature Importance for Random Forest
    if "Random Forest" in trained_models:
        plot_feature_importance(trained_models["Random Forest"], X.columns, os.path.join(results_dir, "feature_importance_rf.png"))

    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(comparison_df.to_string(index=False))
    print("="*50 + "\n")

    if results:
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        logging.info(f"Best Model: {best_model_name}")
        
        # 9. Final Check
        test_urls = ["http://google.com", "http://login-verify-account.com"]
        print("\n--- Quick Prediction Check ---")
        for t_url in test_urls:
            features = extract_features_from_url(t_url)
            features_df = pd.DataFrame([features])
            features_df = features_df[best_model.feature_names_in_]
            pred = best_model.predict(features_df)[0]
            label = "Malicious" if pred == 1 else "Benign"
            print(f"URL: {t_url} -> {label}")
    else:
        logging.error("No models were successfully trained.")

if __name__ == "__main__":
    main()
