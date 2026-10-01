from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

iris = load_iris()

print("Feature names:")
print(iris.feature_names)

print("\nTarget names:")
print(iris.target_names)

print("\nFirst 5 rows of data:")
print(iris.data[:5])

print("\nFirst 5 targets:")
print(iris.target[:5])

print("\nNumber of flowers:")
print(iris.data.shape)

print("\nNumber of classes:")
print(len(iris.target_names))

X = iris.data
y = iris.target

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nFirst X:")
print(X[0])

print("\nFirst y:")
print(y[0])

plt.scatter(X[y==0, 2], X[y==0, 3], label="Setosa")
plt.scatter(X[y==1, 2], X[y==1, 3], label="Versicolor")
plt.scatter(X[y==2, 2], X[y==2, 3], label="Virginica")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset - Petal Length vs Petal Width")
plt.legend()
plt.show()

plt.hist(X[:, 2], bins=10)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Number of Flowers")
plt.title("Distribution of Petal Length")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining data shape:")
print(X_train.shape)
print("\nTesting data shape:")
print(X_test.shape)
print("\nTraining labels shape:")
print(y_train.shape)
print("\nTesting labels shape:")
print(y_test.shape)

print("\nMissing values in X:")
print(np.isnan(X).sum())
print("\nMissing values in X_train:")
print(np.isnan(X_train).sum())
print("\nMissing values inX_test:")
print(np.isnan(X_test).sum())

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nPredicted labels:")
print(y_pred)
print("\nActual labels:")
print(y_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(accuracy)

precision = precision_score(y_test, y_pred, average="weighted")
print("\nPrecision:")
print(precision)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))