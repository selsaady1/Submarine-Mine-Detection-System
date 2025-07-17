#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Mine Versus Rock Project  
# 
# **Author:** Saif Samer Elsaady  
# **Course:** EEE 591 – Python for Rapid Engineering Solutions  
# **Date:** June 12, 2025

# In[3]:


# proj2.py

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from warnings import filterwarnings

# Suppress convergence warnings
filterwarnings('ignore')

# Step 2: Load and Preprocess the Data
# Load sonar data from local path
file_path = r"C:\Users\Saifs\Downloads\sonar_all_data_2.csv"
df = pd.read_csv(file_path, header=None)

# Map labels: 'R' → 1 (rock), 'M' → 2 (mine)
df[60] = df[60].map({'R': 1, 'M': 2})

# Split into features and labels
X = df.iloc[:, :-1].values  # columns 0 to 59 are features
y = df.iloc[:, -1].values   # column 60 is the label


# In[9]:


# Step 2 (continued): Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Step 3: Split the Data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Immediately save a safe copy of the test labels for confusion matrix
y_test_fixed = y_test.copy()

# Step 4: PCA + MLPClassifier Loop
accuracies = []
all_predictions = []

for n_components in range(1, 61):
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(100),
        activation='logistic',
        max_iter=2000,
        alpha=0.00001,
        solver='adam',
        tol=0.0001,
        random_state=42
    )
    model.fit(X_train_pca, y_train)

    # Predict and compute test accuracy
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test_fixed, y_pred)

    # Store results
    accuracies.append(accuracy)
    all_predictions.append(y_pred)

    # Print number of components and corresponding accuracy
    print(f"Components: {n_components:2d}  |  Accuracy: {accuracy:.4f}")


# In[12]:


# Step 5: Evaluate Best Model

# Identify best number of components
best_accuracy = max(accuracies)
best_n = accuracies.index(best_accuracy) + 1  # +1 because index starts at 0
print("\nBest Accuracy: {:.4f} with {} principal components".format(best_accuracy, best_n))

# Re-run PCA and model using best_n components
pca = PCA(n_components=best_n)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

best_model = MLPClassifier(
    hidden_layer_sizes=(100),
    activation='logistic',
    max_iter=2000,
    alpha=0.00001,
    solver='adam',
    tol=0.0001,
    random_state=42
)
best_model.fit(X_train_pca, y_train)
y_best_pred = best_model.predict(X_test_pca)

# Print confusion matrix
cmat = confusion_matrix(y_test_fixed, y_best_pred)
print("\nConfusion Matrix (for {} components):\n".format(best_n), cmat)


# Derive test‐set class counts from cmat
rock_count = cmat[0].sum()   # sum of row 0
mine_count = cmat[1].sum()   # sum of row 1

print(f"\nTest Set Class Distribution: Rocks = {rock_count}, Mines = {mine_count}")


# In[13]:


# Step 6: Plot Accuracy vs Number of Components
plt.figure(figsize=(8,5))
plt.plot(range(1, 61), accuracies, marker='o')
plt.xlabel('Number of PCA Components')
plt.ylabel('Test Accuracy')
plt.title('Accuracy vs. PCA Components')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Print Summary
print("\nSummary of Test Accuracies by Number of Components:")
for idx, acc in enumerate(accuracies, start=1):
    print(f"{idx:2d} components: {acc:.4f}")

print(f"\nBest Accuracy: {best_accuracy:.4f} achieved with {best_n} components")
print("\nConfusion Matrix for Best Model:")
print(cmat)

