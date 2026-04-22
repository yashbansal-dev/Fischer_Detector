# KMeans Hybrid Confusion Matrix Analysis

**Graph File:** `KMeans_Hybrid_cm.png`

## Overview
This is an experimental hybrid approach using unsupervised clustering (KMeans) as a feature engineering step before Logistic Regression.

## Performance Context
*   **Accuracy:** 81.58%
*   **Precision:** 81.10%
*   **Recall:** 81.58%
*   **F1-Score:** 79.36%

## Key Observations
1.  **Lower Resolution:** The confusion matrix shows significantly more "leakage" into the False Positive and False Negative quadrants compared to other models.
2.  **Structural Struggles:** Relying on clusters before classification seems to discard some of the finer lexical signals found in the URLs.

## Conclusion
While an interesting experiment, the visual evidence shows this model is significantly less reliable than our primary tree-based or similarity-based classifiers.
