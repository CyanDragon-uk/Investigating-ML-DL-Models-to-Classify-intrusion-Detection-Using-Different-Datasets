import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# File paths
X_train_file = "X_train.csv"
y_train_file = "y_train.csv"
X_test_file = "X_test.csv"
y_test_file = "y_test.csv"

# Load test data
print("Loading test data...")
X_test = pd.read_csv(X_test_file, low_memory=False)
y_test = pd.read_csv(y_test_file).values.ravel()

# Convert valid numeric columns to float32
for col in X_test.columns:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').astype('float32')

# Replace infinities and clip extreme values
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.fillna(0, inplace=True)
X_test = X_test.clip(-1e6, 1e6)

# Define batch size for chunk processing
chunk_size = 50000  # Adjust based on memory
svm_model = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=42)

# Process training data in chunks
print("Training SVM model in chunks...")
for i, chunk in enumerate(pd.read_csv(X_train_file, chunksize=chunk_size, low_memory=False)):
    y_chunk = pd.read_csv(y_train_file, chunksize=chunk_size, low_memory=False).__next__().values.ravel()

    if chunk.shape[0] != len(y_chunk):
        print(f"⚠ Warning: Mismatched chunk size in batch {i+1} → Features: {chunk.shape[0]}, Labels: {len(y_chunk)}")
        min_size = min(chunk.shape[0], len(y_chunk))
        chunk = chunk.iloc[:min_size, :]
        y_chunk = y_chunk[:min_size]  # Ensure alignment

    print(f" Processing chunk {i+1} with {chunk.shape[0]} rows...")

    # Convert numeric columns properly
    for col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float32')

    # Replace infinities and clip extreme values
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.fillna(0, inplace=True)
    chunk = chunk.clip(-1e6, 1e6)

    # Train on this chunk
    svm_model.partial_fit(chunk, y_chunk, classes=[0, 1])

# Make predictions
print("Making predictions...")
y_pred = svm_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SVM Confusion Matrix (Chunk Training - Fixed)")
plt.show()
