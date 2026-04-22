import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculates Accuracy, Precision, Recall, F1
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    logging.info(f"--- {model_name} Evaluation ---")
    logging.info(f"Accuracy:  {acc:.4f}")
    logging.info(f"Precision: {prec:.4f}")
    logging.info(f"Recall:    {rec:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

def create_comparison_table(results_list):
    """
    Creates and returns a Pandas DataFrame summarizing all model metrics.
    """
    df = pd.DataFrame(results_list)
    # Sort by Accuracy
    df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return df

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plots the confusion matrix using seaborn/matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix for {model_name} saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_model_comparison(comparison_df, save_path):
    """
    Plots a bar chart comparing Accuracy and F1-Score of different models.
    """
    plt.figure(figsize=(10, 6))
    
    # Melt the dataframe for easier plotting with seaborn
    df_melted = comparison_df.melt(id_vars="Model", value_vars=["Accuracy", "F1-Score"], 
                                   var_name="Metric", value_name="Score")
    
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")
    
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path)
    logging.info(f"Performance comparison chart saved to {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    """
    Plots the feature importance for tree-based models.
    """
    if not hasattr(model, 'feature_importances_'):
        logging.warning(f"Model {type(model).__name__} does not have feature_importances_ attribute.")
        return
        
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=feature_names)
    feat_importances = feat_importances.sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8))
    feat_importances.plot(kind='barh', color='skyblue')
    plt.title('Feature Importance - Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_path)
    logging.info(f"Feature importance chart saved to {save_path}")
    plt.close()

def plot_precision_recall_comparison(y_test, trained_models, X_test, save_path):
    """
    Plots Precision-Recall curves for all models in one plot.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(10, 8))
    
    for name, model in trained_models.items():
        try:
            # Check if model supports predict_proba
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                # Fallback for models like LinearSVC without calibration (though our SVM is calibrated)
                y_scores = model.decision_function(X_test)
                
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            avg_p = average_precision_score(y_test, y_scores)
            
            plt.plot(recall, precision, label=f'{name} (AP={avg_p:.2f})')
        except Exception as e:
            logging.warning(f"Could not plot PR curve for {name}: {e}")
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(save_path)
    logging.info(f"Precision-Recall comparison saved to {save_path}")
    plt.close()
