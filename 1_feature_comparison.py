import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Part 1: Original Accuracy Calculation ---

# Load the data
data = pd.read_csv("data/iris_data.csv", header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Separate features and target
X1_sepals = data[['sepal_length', 'sepal_width']] # Sepal features
X2_petals = data[['petal_length', 'petal_width']] # Petal features
y_text = data['species']

# Split the data
train_X1, val_X1, train_y1, val_y1 = train_test_split(X1_sepals, y_text, random_state=1)
train_X2, val_X2, train_y2, val_y2 = train_test_split(X2_petals, y_text, random_state=1)

# Model 1: Using Sepal Features
model1 = DecisionTreeClassifier(random_state=1)
model1.fit(train_X1, train_y1)
train_acc1 = accuracy_score(train_y1, model1.predict(train_X1))
val_acc1 = accuracy_score(val_y1, model1.predict(val_X1))

# Model 2: Using Petal Features
model2 = DecisionTreeClassifier(random_state=1)
model2.fit(train_X2, train_y2)
train_acc2 = accuracy_score(train_y2, model2.predict(train_X2))
val_acc2 = accuracy_score(val_y2, model2.predict(val_X2))

# Print Results
print("--- Feature Performance Comparison ---")
print("\nModel 1: Using Sepal Features")
print(f"Training Accuracy:   {train_acc1:.3f}")
print(f"Validation Accuracy: {val_acc1:.3f}")
print("\nModel 2: Using Petal Features")
print(f"Training Accuracy:   {train_acc2:.3f}")
print(f"Validation Accuracy: {val_acc2:.3f}")
