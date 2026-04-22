# Stacking Classifier Confusion Matrix Analysis

**Graph File:** `Stacking_cm.png`

## Overview
This matrix shows the performance of our Stacking ensemble, which combines the strengths of multiple base models.

## Performance Context
*   **Accuracy:** 97.29%
*   **Precision:** 97.29%
*   **Recall:** 97.29%
*   **F1-Score:** 97.29%

## Key Observations
1.  **High Robustness:** The Stacking model is extremely consistent, showing very few misclassifications across both classes.
2.  **Ensemble Efficiency:** By combining different types of learners, it achieves a performance level nearly identical to Random Forest.

## Conclusion
Stacking is a powerful high-accuracy option, though we currently use Random Forest for the API due to its faster inference time.
