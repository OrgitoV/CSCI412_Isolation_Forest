from sklearn.model_selection import cross_val_score

def overfitting_check(svm, X_train, y_train, scoring = 'f1'):
    # Check for overfitting
    cv_scores = cross_val_score(svm, X_train, y_train, cv = 5, scoring = scoring)
    print(f"SVM Cross-Validation {scoring.upper()} Scores: {cv_scores}")
    print(f"SVM CV Mean {scoring.upper()}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

def distribution_check(y_train_base, y_test_base, predictions):
    # Check class distribution
    print(f"\nTraining set anomaly rate: {y_train_base.mean():.2%}")
    print(f"Test set anomaly rate: {y_test_base.mean():.2%}")
    print(f"SVM prediction anomaly rate: {predictions.mean():.2%}")