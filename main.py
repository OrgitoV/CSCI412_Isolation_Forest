from sklearn.ensemble import RandomForestClassifier as rfc, VotingClassifier
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from create_data import create_isf_data, scores, print_scores
from debugging import overfitting_check, distribution_check
from models import isolationForest, baselineSVM, linearSVM
from graph_data import graph_data, graph_scores

# Run: LinearSCV (baseline), Improved RBF+augmented, Ensemble Voting, Hyperparameter-tuned variant

def main():
    anomaly_split = 0.1
    contamination = (anomaly_split)
    df, features = create_isf_data(data_size = 10000, anomaly_split = anomaly_split)
    true_labels = df['is-anomaly'].values.astype(int) # Cheat-Sheet answers

    graph_data(df, features)

    # Model cannot use certain types of data (strings such as 'ip-address') so we only include numerical
    X = df[features]

    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X, true_labels, test_size=0.2, random_state=42
    )

    # Fit Scaler() on training data only
    scaler_object = scaler()
    X_train_scaled = scaler_object.fit_transform(X_train_base)
    X_test_scaled = scaler_object.transform(X_test_base)

    # Isolation Forest Model and Testing
    isf_model, isf_predictions = isolationForest(contamination, X_train_scaled, X_test_scaled)
    isf_f1, isf_precision, isf_recall, isf_accuracy, isf_confusion_matrix, isf_sensitivity, isf_specificity = scores(y_test_base, isf_predictions)
    print_scores("ISOLATION FOREST STATISTIC SCORES", isf_f1, isf_precision, isf_recall, isf_accuracy, isf_confusion_matrix, isf_sensitivity, isf_specificity)

    """=== BASELINE SVM ==="""

    baseline_svm, baseline_predictions = baselineSVM(X_train_scaled, X_test_scaled, y_train_base)
    baseline_f1, baseline_precision, baseline_recall, baseline_accuracy, baseline_confusion_matrix, baseline_sensitivity, baseline_specificity = scores(y_test_base, baseline_predictions)
    print_scores("BASELINE SVM STATISTIC SCORES", baseline_f1, baseline_precision, baseline_recall, baseline_accuracy, baseline_confusion_matrix, baseline_sensitivity, baseline_specificity)

    # Checks
    distribution_check(y_train_base, y_test_base, baseline_predictions)
    overfitting_check(svm = baseline_svm, X_train = X_train_scaled, y_train = y_train_base, scoring = 'f1')


    """=== IMPROVED: RBF KERNEL SVM ==="""

    # Train RBF SVM on original features with better regularization
    improved_svm, improved_predictions = linearSVM(X_train_scaled, X_test_scaled, y_train_base)

    improved_f1, improved_precision, improved_recall, improved_accuracy, improved_confusion_matrix, improved_sensitivity, improved_specificity = scores(y_test_base, improved_predictions)
    print_scores("IMPROVED SVM (RBF KERNEL) STATISTIC SCORES", improved_f1, improved_precision, improved_recall, improved_accuracy, improved_confusion_matrix, improved_sensitivity, improved_specificity)

    # Checks
    distribution_check(y_train_base, y_test_base, improved_predictions)
    overfitting_check(svm = improved_svm, X_train = X_train_scaled, y_train = y_train_base, scoring = 'f1')

    """=== ENSEMBLE: VOTING CLASSIFIER ==="""

    # Create RandomForest and RBF SVM for ensemble
    rf_model = rfc(n_estimators=100, max_depth=10, class_weight='balanced', random_state=123)
    rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=123, probability=True)
    
    # Voting ensemble: combine RBF SVM with RandomForest (both support predict_proba)
    voting_clf = VotingClassifier(
        estimators=[
            ('rbf_svm', rbf_svm),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    voting_clf.fit(X_train_scaled, y_train_base)
    ensemble_predictions = voting_clf.predict(X_test_scaled)

    ensemble_f1, ensemble_precision, ensemble_recall, ensemble_accuracy, ensemble_confusion_matrix, ensemble_sensitivity, ensemble_specificity = scores(y_test_base, ensemble_predictions)
    print_scores("ENSEMBLE VOTING (RBF SVM + RANDOM FOREST) STATISTIC SCORES", ensemble_f1, ensemble_precision, ensemble_recall, ensemble_accuracy, ensemble_confusion_matrix, ensemble_sensitivity, ensemble_specificity)

    # Checks
    distribution_check(y_train_base, y_test_base, ensemble_predictions)

    # Create comparison chart for all models
    models_dict = {
        'Isolation Forest': {'accuracy': isf_accuracy, 'sensitivity': isf_sensitivity, 'specificity': isf_specificity, 'f1': isf_f1, 'recall': isf_recall},
        'Baseline SVM': {'accuracy': baseline_accuracy, 'sensitivity': baseline_sensitivity, 'specificity': baseline_specificity, 'f1': baseline_f1, 'recall': baseline_recall},
        'Improved RBF SVM': {'accuracy': improved_accuracy, 'sensitivity': improved_sensitivity, 'specificity': improved_specificity, 'f1': improved_f1, 'recall': improved_recall},
        'Ensemble Voting': {'accuracy': ensemble_accuracy, 'sensitivity': ensemble_sensitivity, 'specificity': ensemble_specificity, 'f1': ensemble_f1, 'recall': ensemble_recall}
    }
    graph_scores(models_dict)

if __name__ == "__main__":
    main()