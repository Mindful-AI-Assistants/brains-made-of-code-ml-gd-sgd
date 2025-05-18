


# Stochastic Gradient Descent Linear Regression from Scratch

This code implements a **Stochastic Gradient Descent (SGD)** algorithm to perform linear regression on a multivariate dataset. The code demonstrates how to manually train a linear regression model using SGD, updating weights based on one random data point at a time, which often leads to faster convergence on large datasets.

---

## Step-by-Step Guide with Code

### 1. Import Libraries and Load Data

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from google.colab import drive

# Mount Google Drive to access dataset

drive.mount('/content/drive')

# Load dataset from Excel file

dados = pd.read_excel('/content/drive/MyDrive/class_8-Lasso Regression/Regresao_Lasso_Ridge.xlsx')

# Display first five rows

dados.head()

```

---

### 2. Data Preprocessing

```


# Remove unused 'name' column

dados = dados.drop('name', axis=1)

# Normalize feature columns (good practice but optional)

from sklearn.preprocessing import StandardScaler
normalizador = StandardScaler()

features_to_normalize = ['cylinders', 'displacement', 'horsepower',
'weight', 'acceleration', 'year', 'origin']

dados[features_to_normalize] = normalizador.fit_transform(dados[features_to_normalize])

# Show normalized data preview

dados.head()

```

---

### 3. Add Intercept and Prepare Features and Target

```


# Add intercept column for bias term

dados['intercept'] = 1

# Define independent variables (X) and dependent variable (y)

X = dados[['cylinders', 'displacement', 'horsepower', 'weight',
'acceleration', 'year', 'origin', 'intercept']].values
y = dados['mpg'].values

```

---

### 4. Define Loss Function (Mean Squared Error)

```

def loss_function(X, y, weights, bias):
predictions = np.dot(X, weights) + bias
return np.mean((predictions - y) ** 2)

```

---

### 5. Compute Gradient for a Random Single Data Point

```

def compute_gradient(X, y, weights, bias):
index = random.randint(0, len(X) - 1)
x_example = X[index]
y_example = y[index]
prediction = np.dot(x_example, weights) + bias

    gradient_weights = x_example * (prediction - y_example)
    gradient_bias = prediction - y_example
    
    return gradient_weights, gradient_bias
    ```

---

### 6. Stochastic Gradient Descent Algorithm

```

def stochastic_gradient_descent(X, y, learning_rate, num_iterations, feature_names):
weights = np.zeros(X.shape)
bias = 0

    for i in range(num_iterations):
        gradient_weights, gradient_bias = compute_gradient(X, y, weights, bias)
    
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
    
        if i % 100 == 0:
            loss = loss_function(X, y, weights, bias)
            print(f"Iteration {i}: Loss = {loss:.4f}")
    
    # Map weights to feature names
    weights_with_names = {name: weights[i] for i, name in enumerate(feature_names)}
    weights_with_names['intercept'] = bias
    
    return weights_with_names
    ```

---

### 7. Set Hyperparameters and Run SGD

```

learning_rate = 0.01
num_iterations = 100000
feature_names = ['cylinders', 'displacement', 'horsepower', 'weight',
'acceleration', 'year', 'origin']

final_weights = stochastic_gradient_descent(X, y, learning_rate, num_iterations, feature_names)

print("\nFinal Weights:")
for feature, weight in final_weights.items():
print(f"{feature:>12}: {weight:.4f}")

```

---

## Suggested Project Name

**"Stochastic Gradient Descent Linear Regression Engine"**

---

## Brief Explanation

This code implements **Stochastic Gradient Descent (SGD)** to train a linear regression model from scratch. Unlike batch gradient descent that uses all data points to update weights, SGD updates weights using one randomly selected example at each iteration. This often leads to faster convergence and better scalability for large datasets.

The process includes:
- Loading and normalizing the dataset
- Adding an intercept term for bias
- Iteratively updating weights and bias based on single random samples
- Tracking the mean squared error loss during training
- Outputting the final learned weights for each feature and the intercept

This hands-on implementation helps understand the mechanics behind SGD and linear regression without relying on high-level libraries.



