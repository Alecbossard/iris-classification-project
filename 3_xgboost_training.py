import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load and prepare the data
data = pd.read_csv("data/iris_data.csv", header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
X = data.drop('species', axis=1)
y = LabelEncoder().fit_transform(data['species'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

# Create and train the model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=0, eval_metric='mlogloss')
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=False
)

# Plot learning curves
results = model.evals_result()
plt.figure(figsize=(8, 5))
plt.plot(results['validation_0']['mlogloss'], label='Train Logloss')
plt.plot(results['validation_1']['mlogloss'], label='Validation Logloss')
plt.xlabel("Number of Trees")
plt.ylabel('Log Loss')
plt.title('XGBoost Learning Curves')
plt.legend()
plt.show()

# Print final accuracy
y_pred = model.predict(X_valid)
print(f"\nFinal XGBoost Model Accuracy: {accuracy_score(y_valid, y_pred):.4f}")