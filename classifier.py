import numpy as np
from sklearn.ensemble import IsolationForest as isf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


def add_isolation_feature(X, y, test_size = 0.2, random_state = 43):
    """Train ISO Forest on training set only to avoid data leakage"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state)
    iso = isf(contamination = 0.05, random_state = random_state)
    iso.fit(X_train)

    train_scores = iso.decision_function(X_train).reshape(-1, 1)
    test_scores = iso.decision_function(X_test).reshape(-1, 1)

    X_train_new = np.hstack((X_train, train_scores))
    X_test_new = np.hstack((X_test, test_scores))
    return X_train_new, X_test_new, y_train, y_test, iso


def run_classification(X, y, model_type="svm"):

    X_train, X_test, y_train, y_test, iso = add_isolation_feature(X, y)

    if model_type == "svm":
        clf = SVC(class_weight = 'balanced')
    elif model_type == "tree":
        clf = DecisionTreeClassifier()
    else:
        raise ValueError("Invalid model_type")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== WITH Isolation Forest Feature ===")
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1')
    print(f"Cross-validation F1 scores: {scores}")
    print(f"Mean F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return clf, iso


def run_baseline(X, y, model_type="svm"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "svm":
        clf = SVC(class_weight = 'balanced')
    elif model_type == "tree":
        clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== BASELINE (No Isolation Forest) ===")
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1')
    print(f"Cross-validation F1 scores: {scores}")
    print(f"Mean F1: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return clf