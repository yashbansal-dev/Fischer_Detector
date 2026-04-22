from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import logging

def get_logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

def get_knn(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

def get_svm():
    # Use LinearSVC for speed on 650k rows. 
    # Wrap in CalibratedClassifierCV if probabilities are needed for Stacking/Voting.
    from sklearn.calibration import CalibratedClassifierCV
    base_svm = LinearSVC(max_iter=2000, random_state=42, dual=False)
    # Stacking/Voting with 'soft' requires probabilities. LinearSVC doesn't have them natively.
    return CalibratedClassifierCV(base_svm, cv=3)

def get_random_forest():
    # Increase n_jobs to use all cores
    return RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)

def get_kmeans(n_clusters=2):
    """
    For unsupervised learning analysis. K-Means finds clusters before applying labels.
    """
    return KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')

def get_voting_classifier():
    """
    Voting Classifier (Logistic Regression + SVM + Random Forest)
    """
    estimators = [
        ('lr', get_logistic_regression()),
        ('rf', get_random_forest()),
        ('svm', get_svm())
    ]
    # Hard voting means majority rules; soft uses probabilities.
    return VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)

def get_stacking_classifier():
    """
    Stacking (KNN + SVM -> Random Forest as meta-model)
    """
    estimators = [
        ('knn', get_knn()),
        ('svm', get_svm())
    ]
    meta_model = get_random_forest()
    return StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=3, n_jobs=-1)

def get_kmeans_hybrid_pipeline(n_clusters=5):
    """
    Hybrid pipeline (K-Means clustering + supervised model)
    K-Means acts as a transformer, converting raw features to distances to cluster centers.
    Logistic Regression then trains on those distances.
    """
    return Pipeline([
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')),
        ('lr', get_logistic_regression())
    ])

def train_model(model_name, model, X_train, y_train):
    """
    Trains a given model and returns it.
    """
    logging.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logging.info(f"{model_name} trained successfully.")
    return model
