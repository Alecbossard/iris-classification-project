import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Load the data
data = pd.read_csv("data/iris_data.csv", header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
X = data[['petal_length', 'petal_width']].values
y_encoded = LabelEncoder().fit_transform(data['species'])

# Prepare the grid for plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Models to compare
models = [
    ("Decision Tree", DecisionTreeClassifier(max_depth=3, random_state=0)),
    ("XGBoost", XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=0, eval_metric='mlogloss'))
]

# Create the plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, (name, model) in zip(axes, models):
    model.fit(X, y_encoded)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=data["species"], palette="Set2", ax=ax)
    ax.set_title(f"Decision Boundaries - {name}")
    ax.set_xlabel("Petal length")
    ax.set_ylabel("Petal width")

plt.tight_layout()
plt.show()