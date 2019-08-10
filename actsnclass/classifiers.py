from sklearn.ensemble import RandomForestClassifier


def random_forest(train_features, train_labels, test_features):
    """Random Forest classifier."""

    clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    prob = clf.predict_proba(test_features)

    return predictions, prob