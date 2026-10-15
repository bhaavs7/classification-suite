from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.classifiers import get_classifier

def main():
    print("=== Classification Suite Benchmark ===")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_names = ["logistic_regression", "random_forest", "xgboost", "knn", "svm"]

    for name in model_names:
        model = get_classifier(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model: {name:<22} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()