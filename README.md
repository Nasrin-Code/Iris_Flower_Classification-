# Iris Flower Classification

## Project Overview
This project uses Machine Learning to classify Iris flowers into three species: Setosa, Versicolor, and Virginica, based on sepal and petal measurements.

## Objective
Build a classification model that predicts the species of an Iris flower using its measurements.

## Dataset
- **Source:** Iris dataset from scikit-learn
- **Total samples:** 150
- **Features:** 4
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Classes:** Setosa, Versicolor, Virginica

## Technologies Used
- Python
- NumPy
- Matplotlib
- scikit-learn

## Project Workflow
1. Load the Iris dataset.
2. Explore the data and feature names.
3. Visualize the data using a scatter plot and histogram.
4. Split the dataset into training and testing sets.
5. Check for missing values.
6. Train a Logistic Regression classifier.
7. Predict flower species using the test data.
8. Evaluate the model using accuracy, precision, a confusion matrix, and a classification report.

## Model
**Logistic Regression**

The dataset is split into:
- Training set: 80% (120 samples)
- Testing set: 20% (30 samples)

## Results
Results from the current train-test split:

- **Accuracy:** 100%
- **Weighted precision:** 100%
- **Correct predictions:** 30 out of 30

The confusion matrix and classification report are printed by the Python script.

*Note: These results apply to this particular test split and do not guarantee the same performance on new data.*

## How to Run

1. Install Python.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python iris_classification.py
   ```

## Project Structure

```text
Iris_Flower_Classification/
│
├── iris_classification.py
├── requirements.txt
└── README.md
```

## Internship
QSkill AI & Machine Learning Internship

**Task:** Iris Flower Classification