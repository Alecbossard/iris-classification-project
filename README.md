# 🌸 Iris Dataset Analysis Project

This project contains a series of Python scripts that explore the Iris dataset and compare different classification models.

## Project Structure

The analysis is divided into several numbered scripts that can be run in order to follow the workflow:

1.  `1_feature_comparison.py`: Compares the performance of a decision tree using either sepal or petal features.
2.  `2_data_visualization.py`: Generates exploratory plots (pairplot, boxplot) and calculates feature importances.
3.  `3_xgboost_training.py`: Trains an XGBoost model and displays its learning curve.
4.  `4_decision_boundary_visualization.py`: Displays and compares the decision boundaries of a Decision Tree and XGBoost.

## How to Use

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the scripts one by one:
    ```bash
    python 1_feature_comparison.py
    python 2_data_visualization.py
    # and so on...
    ```
