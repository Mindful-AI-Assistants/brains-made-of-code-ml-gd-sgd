
## Statement:

#### From the Base Imoveis.xlsx build a Regression based on the Descending Gradient and the Stochastic Descending Gradient. 
#### Use 0.01 as the learning rate for both cases. Evaluate the weights obtained in each of the models.


<br>

```markdown
# Regression Analysis on Real Estate Dataset using Batch and Stochastic Gradient Descent

## Project Overview

This notebook develops and compares two linear regression models for predicting real estate values (`Valor`) based on property features. The models are trained using two optimization techniques: **Batch Gradient Descent** and **Stochastic Gradient Descent**. The goal is to understand the effectiveness of each method in terms of convergence, accuracy, and interpretability when applied to the real estate dataset.

---

## Dataset Description

The dataset (`Base Imoveis.xlsx`) contains information about properties, with the following relevant columns:

- **Valor**: Target variable. The market value of the property.
- **Área**: The area (size) of the property.
- **Idade**: The age of the property.
- **Energia**: Energy consumption or rating of the property.

---

## Methodology

1. **Exploratory Data Analysis (EDA):**
   - Statistical summary and visualization of the data.
   - Checking for missing values and data distribution.

2. **Data Preparation:**
   - Handling missing values by imputing with the mean.
   - Feature normalization for improved convergence.

3. **Model Training:**
   - Splitting the data into training and test sets.
   - Implementing Batch Gradient Descent and Stochastic Gradient Descent from scratch, with early stopping for efficiency.
   - Comparing convergence using loss curves.

4. **Model Evaluation:**
   - Assessing performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score on the test set.
   - Visualizing predicted vs. actual values.
   - Comparing the learned weights for interpretability.

---

## Expected Outcomes

- **Performance Comparison:** Direct comparison of the two optimization algorithms in terms of speed, stability, and accuracy.
- **Feature Insights:** Understanding which features most influence property value, based on the learned weights.
- **Practical Skills:** Hands-on experience with implementing and evaluating linear regression with custom optimization routines.

---
```

```python
# Cell 1: Imports and Random Seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)  # For reproducibility
```

```python
# Cell 2: Load Dataset
# Load the Imoveis dataset
data = pd.read_excel('Base Imoveis.xlsx')
data.head()
```

```python
# Cell 3: Exploratory Data Analysis
# Display basic statistics and check for missing values
print(data.describe())
print("\nMissing values per column:\n", data.isnull().sum())

# Visualize feature relationships
pd.plotting.scatter_matrix(data, figsize=(10, 7))
plt.suptitle('Feature Relationships')
plt.show()
```

```python
# Cell 4: Handling Missing Values
# Fill missing values with the mean (if any)
data = data.fillna(data.mean())
```

```python
# Cell 5: Feature Selection
# Define features and target
target_column = 'Valor'
features = ['Área', 'Idade', 'Energia']
```

```python
# Cell 6: Train-Test Split
# Split the data for model evaluation
X = data[features].values
y = data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

```python
# Cell 7: Normalization
# Normalize features for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add intercept column
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
feature_names = features + ['intercept']
```

```python
# Cell 8: Loss Function
# Mean Squared Error loss function
def mse_loss(X, y, weights):
    predictions = np.dot(X, weights)
    return np.mean((predictions - y) ** 2)
```

```python
# Cell 9: Batch Gradient Descent with Early Stopping
def batch_gradient_descent(X, y, learning_rate=0.01, num_iterations=10000, tol=1e-6, verbose=True):
    weights = np.zeros(X.shape[1])
    losses = []
    prev_loss = float('inf')
    for i in range(num_iterations):
        predictions = np.dot(X, weights)
        gradient = np.dot(X.T, (predictions - y)) / len(y)
        weights -= learning_rate * gradient
        if i % 100 == 0:
            loss = mse_loss(X, y, weights)
            losses.append(loss)
            if verbose:
                print(f'Batch GD Iteration {i}: Loss = {loss}')
            if abs(prev_loss - loss) < tol:
                print(f"Early stopping at iteration {i}, loss: {loss}")
                break
            prev_loss = loss
    return weights, losses
```

```python
# Cell 10: Stochastic Gradient Descent with Early Stopping
def stochastic_gradient_descent(X, y, learning_rate=0.01, num_iterations=10000, tol=1e-6, verbose=True):
    weights = np.zeros(X.shape[1])
    losses = []
    prev_loss = float('inf')
    n_samples = X.shape[0]
    for i in range(num_iterations):
        idx = np.random.randint(0, n_samples)
        x_i = X[idx]
        y_i = y[idx]
        prediction = np.dot(x_i, weights)
        gradient = (prediction - y_i) * x_i
        weights -= learning_rate * gradient
        if i % 100 == 0:
            loss = mse_loss(X, y, weights)
            losses.append(loss)
            if verbose:
                print(f'SGD Iteration {i}: Loss = {loss}')
            if abs(prev_loss - loss) < tol:
                print(f"Early stopping at iteration {i}, loss: {loss}")
                break
            prev_loss = loss
    return weights, losses
```

```python
# Cell 11: Train Models
# Train both models
batch_weights, batch_losses = batch_gradient_descent(X_train, y_train, learning_rate=0.01, num_iterations=10000)
sgd_weights, sgd_losses = stochastic_gradient_descent(X_train, y_train, learning_rate=0.01, num_iterations=10000)
```

```python
# Cell 12: Evaluation on Test Set
# Predict on test data
y_pred_batch = np.dot(X_test, batch_weights)
y_pred_sgd = np.dot(X_test, sgd_weights)

# Compute evaluation metrics
print("Batch Gradient Descent Metrics:")
print("MSE:", mean_squared_error(y_test, y_pred_batch))
print("MAE:", mean_absolute_error(y_test, y_pred_batch))
print("R2:", r2_score(y_test, y_pred_batch))

print("\nStochastic Gradient Descent Metrics:")
print("MSE:", mean_squared_error(y_test, y_pred_sgd))
print("MAE:", mean_absolute_error(y_test, y_pred_sgd))
print("R2:", r2_score(y_test, y_pred_sgd))
```

```python
# Cell 13: Compare Weights
# Tabulate and compare the learned weights
weights_df = pd.DataFrame({
    'Feature': feature_names,
    'Batch_GD_Weight': batch_weights,
    'SGD_Weight': sgd_weights
})
weights_df
```

```python
# Cell 14: Plot Loss Curves
# Visualize convergence of both algorithms
plt.plot(np.arange(0, len(batch_losses)*100, 100), batch_losses, label='Batch Gradient Descent')
plt.plot(np.arange(0, len(sgd_losses)*100, 100), sgd_losses, label='Stochastic Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('MSE Loss')
plt.title('Loss Curve Comparison')
plt.legend()
plt.show()
```

```python
# Cell 15: Prediction Visualization
# Visualize predicted vs. actual values
plt.scatter(y_test, y_pred_batch, alpha=0.7, label='Batch GD')
plt.scatter(y_test, y_pred_sgd, alpha=0.7, label='SGD')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Valor')
plt.ylabel('Predicted Valor')
plt.title('Actual vs. Predicted Valor')
plt.legend()
plt.show()
```



