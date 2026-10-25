"""
DecodeLabs | Artificial Intelligence | Project 2
Data Classification Using AI (Iris + K-Nearest Neighbors)

Author : Aakarsh Kumar
Batch  : 2026

Pipeline (IPO framework):
    INPUT   -> load Iris dataset, feature scaling (StandardScaler)
    PROCESS -> shuffled 80/20 train-test split, KNN algorithm
    OUTPUT  -> accuracy, confusion matrix, precision / recall / F1 score

Install once:  pip install scikit-learn matplotlib
Run:           python iris_classification.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
K = 5


def main() -> None:
    # ------------------------------------------------------------------
    # 1. INPUT: load and understand the dataset
    # ------------------------------------------------------------------
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names

    df = pd.DataFrame(X, columns=iris.feature_names)
    df["species"] = [class_names[i] for i in y]

    print("=" * 60)
    print("STEP 1: DATASET OVERVIEW")
    print("=" * 60)
    print(f"Samples   : {X.shape[0]}")
    print(f"Features  : {X.shape[1]} -> {list(iris.feature_names)}")
    print(f"Classes   : {len(class_names)} -> {list(class_names)}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nClass balance:")
    print(df["species"].value_counts().to_string())
    print("\nStatistical summary:")
    print(df.describe().round(2).to_string())

    # ------------------------------------------------------------------
    # 2. PROCESS: shuffled train-test split (80% / 20%)
    #    stratify keeps the class balance equal in both sets
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("=" * 60)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Feature scaling (the "Gatekeeper Rule")
    #    Fit on TRAIN only, then transform both -> no data leakage
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Tuning: compare different K values (find "the elbow")
    # ------------------------------------------------------------------
    k_values = list(range(1, 26))
    error_rates = []
    for k in k_values:
        temp_model = KNeighborsClassifier(n_neighbors=k)
        temp_model.fit(X_train_scaled, y_train)
        error_rates.append(1 - accuracy_score(y_test, temp_model.predict(X_test_scaled)))
    best_k = k_values[error_rates.index(min(error_rates))]

    print("\n" + "=" * 60)
    print("STEP 3: CHOOSING K")
    print("=" * 60)
    print(f"Lowest test error occurs at K = {best_k} (error = {min(error_rates):.3f})")
    print(f"Using K = {K} as specified in the project blueprint.")

    # ------------------------------------------------------------------
    # 5. Train the model: Instantiate -> Fit -> Predict
    # ------------------------------------------------------------------
    model = KNeighborsClassifier(n_neighbors=K)   # INSTANTIATE
    model.fit(X_train_scaled, y_train)            # FIT
    predictions = model.predict(X_test_scaled)    # PREDICT

    # ------------------------------------------------------------------
    # 6. OUTPUT: validation metrics
    # ------------------------------------------------------------------
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "=" * 60)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy         : {acc:.2%}")
    print(f"F1 Score (weight): {f1:.4f}")
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=class_names, digits=3))

    # ------------------------------------------------------------------
    # 7. Predict a brand-new flower (completely unseen data)
    # ------------------------------------------------------------------
    new_flower = [[5.9, 3.0, 5.1, 1.8]]   # sepal L, sepal W, petal L, petal W (cm)
    new_scaled = scaler.transform(new_flower)
    result = model.predict(new_scaled)[0]
    print("=" * 60)
    print("STEP 5: PREDICTING A NEW FLOWER")
    print("=" * 60)
    print(f"Measurements {new_flower[0]} -> predicted species: {class_names[result]}")

    # ------------------------------------------------------------------
    # 8. Charts (saved as PNG and shown on screen)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"Confusion Matrix (K={K}, accuracy {acc:.1%})")

    axes[1].plot(k_values, error_rates, marker="o", color="#1f3a5f")
    axes[1].axvline(best_k, color="#e8590c", linestyle="--", label=f"Best K = {best_k}")
    axes[1].set_xlabel("K value")
    axes[1].set_ylabel("Error rate")
    axes[1].set_title("Tuning the Engine: Choosing K")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nChart saved as results.png")
    plt.show()


if __name__ == "__main__":
    main()