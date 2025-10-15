import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load the data
data = pd.read_csv("data/iris_data.csv", header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
y = data['species']

# Plot 1: Pairplot
sns.pairplot(data, hue='species')
plt.suptitle("Cross-Variable Visualization of the Iris Dataset", y=1.02)
plt.show()

# Plot 2: Boxplot
sns.boxplot(data=data, x='species', y='petal_length')
plt.title("Distribution of Petal Length by Species")
plt.show()

# Analysis 3: Feature Importance
model_full = DecisionTreeClassifier(random_state=1)
model_full.fit(data.drop('species', axis=1), y)
importances = pd.Series(model_full.feature_importances_, index=data.columns[:-1])

print("\n--- Feature Importances (according to Decision Tree) ---")
print(importances.sort_values(ascending=False))