
## Gradiente Descendent Linear Regression from Scratch

This code implements a **Gradient Descent algorithm** from scratch to perform **linear regression** on a dataset. The goal is to find the best-fitting linear model that predicts a target variable (`mpg`) based on several input features such as `cylinders`, `displacement`, `horsepower`, and others.

### Key Points:

- **Data Loading:** The dataset is loaded from an Excel file stored on Google Drive.
- **Preprocessing:** Irrelevant columns are removed, and the features are normalized to improve the training process.
- **Intercept Handling:** An intercept term (bias) is explicitly added to the model to allow the regression line to fit the data better.
- **Gradient Descent:** The algorithm iteratively updates model weights and intercept by minimizing the Mean Squared Error (MSE) loss function.
- **Output:** After a fixed number of iterations, the final weights (coefficients) for each feature and the intercept are printed, representing the learned linear relationship.

In summary, this code demonstrates how to manually implement and train a linear regression model using gradient descent, providing insight into the underlying mechanics of machine learning algorithms.


### Gradiente Descendet Algorithmic - Intercept

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
dados = pd.read_excel('/content/drive/MyDrive/class_8-Lasso Regression/Regresao_Lasso_Ridge.xlsx')
\#Exibe as cinco primeiras linhas
dados.head()
\#Tirando a coluna Name que não será usada
dados = dados.drop('name',axis=1)
\#Normalizando os dados - não é obrigatório mas é uma boa prática
from sklearn.preprocessing import StandardScaler
normalizador = StandardScaler()

dados[['cylinders', 'displacement', 'horsepower','weight', 'acceleration', 'year', \
'origin']] = normalizador.fit_transform(dados[['cylinders','displacement','horsepower',\
'weight','acceleration','year', 'origin']])
dados.head()

# Adicionar um intercepto

dados['intercept'] = 1

# Definir as variáveis independentes (X) e a variável dependente (y)

X = dados[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'intercept']]
y = dados['mpg']

# Conversão dos dados para matrizes NumPy

X = X.values
y = y.values

# Função de perda (MSE)

def loss_function(X, y, weights, bias):
predictions = np.dot(X, weights) + bias
return np.mean((predictions - y) ** 2)

# Gradiente da função de perda

def compute_gradient(X, y, pesos):
predictions = np.dot(X, pesos)
gradient = np.dot(X.T, (predictions - y)) / len(y)
return gradient

# Algoritmo de gradiente descendente

def gradient_descent(X, y, learning_rate, num_iterations, feature_names):
\# Inicialização dos pesos e do viés
weights = np.zeros(X.shape[1])
bias = 0

    for i in range(num_iterations):
        # Cálculo das previsões
        predictions = np.dot(X, weights) + bias
    
        # Cálculo do gradiente
        gradient_weights = np.dot(X.T, (predictions - y)) / len(y)
        gradient_bias = np.mean(predictions - y)
    
        # Atualização dos pesos e do viés
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
    
        # Avaliação da função de perda
        if i % 100 == 0:
            loss = loss_function(X, y, weights, bias)
            print(f"Iteração {i}: Perda = {loss}")
    
    # Preparar pesos finais com nomes das variáveis
    weights_with_names = {}
    for i, name in enumerate(feature_names):
        weights_with_names[name] = weights[i]
    weights_with_names['intercept'] = bias
    
    return weights_with_names
    
# Taxa de aprendizado e número de iterações

learning_rate = 0.01
num_iterations = 100000

# Definir os nomes das variáveis independentes

feature_names = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']

# Aplicação do gradiente descendente

final_weights = gradient_descent(X, y, learning_rate, num_iterations, feature_names)
print("Pesos finais:", final_weights)

Here's a well-structured README.md for your Gradient Descent implementation with proper Markdown formatting and code presentation:

```markdown
# Gradient Descent Linear Regression from Scratch

Implementation of a custom gradient descent algorithm for multivariate linear regression with intercept handling. This project demonstrates core machine learning concepts through manual implementation.

## Code Implementation

### 1. Data Preparation & Mounting Drive
```

import numpy as np
import pandas as pd
from google.colab import drive

drive.mount('/content/drive')
dados = pd.read_excel('/content/drive/MyDrive/class_8-Lasso Regression/Regresao_Lasso_Ridge.xlsx')
dados.head()

```

### 2. Data Preprocessing
```


# Remove unused column

dados = dados.drop('name', axis=1)

# Normalize features

from sklearn.preprocessing import StandardScaler
normalizador = StandardScaler()

features_to_normalize = ['cylinders', 'displacement', 'horsepower',
'weight', 'acceleration', 'year', 'origin']
dados[features_to_normalize] = normalizador.fit_transform(dados[features_to_normalize])

# Add intercept column

dados['intercept'] = 1

```

### 3. Dataset Configuration
```


# Define features and target

X = dados[['cylinders', 'displacement', 'horsepower', 'weight',
'acceleration', 'year', 'origin', 'intercept']].values
y = dados['mpg'].values

```

### 4. Core Algorithm Components
```

def loss_function(X, y, weights, bias):
"""Calculate Mean Squared Error"""
predictions = np.dot(X, weights) + bias
return np.mean((predictions - y) ** 2)

def gradient_descent(X, y, learning_rate, num_iterations, feature_names):
"""Custom gradient descent implementation"""
weights = np.zeros(X.shape[^1])
bias = 0

    for i in range(num_iterations):
        predictions = np.dot(X, weights) + bias
        gradient_weights = np.dot(X.T, (predictions - y)) / len(y)
        gradient_bias = np.mean(predictions - y)
        
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
        
        if i % 100 == 0:
            loss = loss_function(X, y, weights, bias)
            print(f"Iteration {i}: Loss = {loss:.4f}")
    
    return {'weights': dict(zip(feature_names, weights)), 'intercept': bias}
    ```

### 5. Execution Parameters
```


# Hyperparameters

LEARNING_RATE = 0.01
ITERATIONS = 100000
FEATURE_NAMES = ['cylinders', 'displacement', 'horsepower',
'weight', 'acceleration', 'year', 'origin']

# Run algorithm

final_params = gradient_descent(X, y, LEARNING_RATE, ITERATIONS, FEATURE_NAMES)
print("\nFinal Parameters:")
for param, value in final_params['weights'].items():
print(f"{param:>12}: {value:.4f}")
print(f"{'intercept':>12}: {final_params['intercept']:.4f}")

```

## Suggested Name
**"Gradient Descent Regression Engine with Intercept Handling"**

---

## Key Features
- **Complete implementation** from data loading to model training
- **Proper feature normalization** using StandardScaler
- **Custom loss tracking** with MSE calculation
- **Interpretable parameter output** with feature names
- **Colab integration** for cloud execution

## Usage Instructions
1. Mount Google Drive in Colab
2. Adjust file path in `pd.read_excel()`
3. Run cells sequentially
4. Monitor loss output every 100 iterations
5. Inspect final parameters

*Note: Requires numpy, pandas, and sklearn for preprocessing*
```

This implementation provides:

- Clear section headers for easy navigation
- Syntax-highlighted code blocks
- Step-by-step execution flow
- Complete parameter reporting
- Hyperparameter documentation

The name emphasizes both the algorithmic approach (gradient descent) and special feature handling (intercept management), making it descriptive and technically precise.


