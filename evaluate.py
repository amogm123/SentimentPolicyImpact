from sklearn.metrics import accuracy_score, f1_score, classification_report

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return acc, f1, report