from sklearn.ensemble import IsolationForest as isf, RandomForestClassifier as rfc, VotingClassifier
from sklearn.svm import LinearSVC

def isolationForest(contamination, X_train, X_test):
    isf_model = isf(n_estimators = 500, max_samples = 1024, contamination = contamination, random_state = 123)
    isf_model.fit(X_train)
    predictions = (isf_model.predict(X_test) == -1).astype(int)

    return isf_model, predictions

def baselineSVM(X_train, X_test, y_train, C = 1.0, loss = 'squared_hinge', class_weight = 'balanced', random_state = 123, max_iter = 3000):
    baseline_svm = LinearSVC(C = C, loss = loss, class_weight = class_weight, random_state = random_state, max_iter = max_iter, dual = False)
    baseline_svm.fit(X_train, y_train)
    predictions = baseline_svm.predict(X_test)

    return baseline_svm, predictions