
<br>

#  <p align="center"> 🧠  [Brain]() Made of Code, [Created]() with [Heart]() ❤︎
### <p align="center"> [From Regression to Neural Nets](): Learning with Gradient Descent & Beyond 

<br><br>


#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-Mindful%20AI%20%20Assistants-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)

<br><br>


<!--Brain - ART GIF -->

<p align="center">
  <img src="https://github.com/user-attachments/assets/20470843-6f04-4e20-81d5-7aa918ca9a2b" width="450"/>
</p>

<!--Confidentiality statement -->




<!--Confidentiality statement -->

<br><br>

#


### ⚠️ Heads Up 

* Projects and deliverables may be made [publicly available]() whenever possible.

* The course prioritizes [**hands-on practice**]() with real data in consulting scenarios.

* All activities comply with the [**academic and ethical guidelines of PUC-SP**]().

* [**Confidential information**]() from this repository remains private in [private repositories]().

#  

<br><br><br>

<!-- End -->


This project provides a **comprehensive and hands-on guide to Machine Learning and Artificial Neural Networks (ANNs)**, combining **theory, Python code, and visualizations**.

<BR>


[It starts from the ground up]():


* **Historical & Mathematical Background:** Full explanations of the theory behind algorithms, including derivations, historical context, and all key formulas. Formulas are provided in LaTeX, ready to copy-paste for documentation or reports.

* **Foundations:** understanding tensors as the core data structure in deep learning.

* **Datasets:** loading and working with real datasets online (e.g., MNIST) for experimentation.

* **Regression & Optimization:** training models with Batch Gradient Descent, Stochastic Gradient Descent, and Mini-batch GD.

* **Regularization:** applying techniques like Elastic Net to reduce overfitting and improve generalization.

* **Model Evaluation:** exploring accuracy metrics, detecting and handling outliers, and understanding error analysis.

* **Neural Networks:** building progressively — from the artificial neuron model to Multilayer Perceptrons (MLP), forward and backward propagation, and optimization improvements like Momentum.


<br><br>


[The repository provides]():

* **Step-by-step explanations** before each implementation.

* **Python code split into reproducible cells**, ready for Jupyter Notebook or Google Colab.

* **Practical examples and visualizations** to illustrate convergence, performance, overfitting behavior, and accuracy measurement.

<br>

Ideal for [**beginners**]() and [**intermediate**]() learners looking to build a solid foundation in [**machine learning optimization algorithms**](), and who want to [**go beyond running code**]() to truly understand the underlying concepts of ***optimization***, ***evaluation***, and [**neural networks**]().



<br><br>


<!--Neural Net Nodes GIF -->

<p align="center">
  <img src="https://github.com/user-attachments/assets/c1d1a6ae-cf3d-4a71-871f-461928012002" />

<!--End-->


<br><br>



# I - [Artificial Neural Networks – From Perceptron to Modern Learning Algorithms]() 

This repository provides a **hands-on and structured overview** of **Artificial Neural Networks (ANNs)** — starting from the **Perceptron**, moving through the **Multilayer Perceptron (MLP)**, and exploring the **learning algorithms** that power Machine Learning, such as **Gradient Descent** and its variations.

The content is inspired by the official theoretical material *[Decreasing-Gradient.pdf](./Decreasing-Gradient.pdf)* from the **Undergraduate Program in Humanistic AI and Data Science at PUC São Paulo, Brazil**.

<br>

## [Motivation]()

The human brain processes information in a **nonlinear, adaptive, and massively parallel** way — very different from how conventional computers work.

For example, the brain can **recognize a familiar face in milliseconds**, even in a completely new environment. Meanwhile, traditional computers may take much longer to solve much simpler problems.

Taking inspiration from biology, **Artificial Neural Networks** are computational models designed to **learn from data, adapt through experience, and replicate human-like problem solving**.


<br>

## [Historical Context]()

- **McCulloch & Pitts (1943):** Introduced the first neural network models.

- **Hebb (1949):** Developed the basic model of self-organization.

- **Rosenblatt (1958):** Introduced the perceptron, a supervised learning model.

- **Hopfield (1982), Rumelhart, Hinton & Williams:** Revived the field with symmetric networks for optimization and the backpropagation method.
  
<br>

## [Artificial Neuron Mode]()

<br>

### Each artificial neuron receives input signals $X_1, X_2, ..., X_p$ (binary or real values), each multiplied by a weight $w_1, w_2, ..., w_p$ (real values). The neuron computes a weighted sum (activity level):

<br>

$$
\Huge
a = w_1 X_1 + w_2 X_2 + \cdots + w_p X_p
$$

<br><br>

```latex
\a = w_1 X_1 + w_2 X_2 + \cdots + w_p X_p\
```

<br><br>
  
#

### <p align="center"> [The output y is determined by an activation function, such as:]()

<br>

$$
\Huge
y =
\begin{cases}
1, & \text{if } a \geq t \\
0, & \text{if } a < t
\end{cases}
$$

<br><br>

```latex
y =
\begin{cases}
1, & \text{if } a \geq t \\
0, & \text{if } a < t
\end{cases}
```
  
<br><br>


## [Key Benefits of ANNs]()

- Adaptability through learning
- Ability to operate with partial knowledge
- Fault tolerance
- Generalization
- Contextual information processing
- Input-output mapping

<br>

## [Application Areas]()

- Pattern classification
- Clustering/categorization
- Function approximation
- Prediction
- Optimization
- Content-addressable memory
- Control systems

  <br>

## [Learning Process]()

ANNs operate in two main phases:

1. **Training Phase:** The network learns by adjusting its free parameters (weights) to perform a specific function.
2. **Application Phase:** The trained network is used for its intended purpose (e.g., pattern or image classification).

<br>

## [The learning process involves:]()

1. Stimulation by the environment (input).
2. Modification of free parameters (weights) as a result.
3. The network responds differently due to internal changes.

Learning is governed by a set of pre-established rules (learning algorithm) and a learning paradigm (model).

<br>

## [Error Correction Learning]()

The output of neuron $k$ at iteration $n$ is $y_k(n)$, and the desired response is $d_k(n)$. The error signal is:

<br>

$$
\Huge
e_k(n) = d_k(n) - y_k(n)
$$

<br><br>
  
#

### [The goal is to minimize the cost function (performance index):]()

<br>

$$
\Huge
E(n) = \frac{1}{2} e_k^2(n)
$$

<br><br>
  
#

### [Weights are updated as:]()

<br>

$$
\Huge
w_{kj}(n+1) = w_{kj}(n) + \Delta w_{kj}(n)
$$

<br><br>
  


# [The Perceptron]()

The perceptron, proposed by Rosenblatt (1958), is the simplest type of ANN. It uses supervised learning and error correction to adjust the weight vector. For a perceptron with two inputs and a bias:

- The bias allows the threshold value in the activation function to be set, and is updated like any other weight.

<br>

## [Nonlinearities and Activation Functions]()

- Nonlinearities are inherent in most real-world problems.

- Incorporated through nonlinear activation functions (e.g., sigmoid, tanh) and multiple layers.

- MLPs use sigmoid functions in hidden layers and linear functions in the output layer.

<br>

## [MLP (MultiLayer Perceptron)]()

- Composed of neurons with nonlinear activation functions in intermediate (hidden) layers.

- Only the output layer receives a desired output during training.

- The error for hidden layers is estimated by the effect they cause on the output error (backpropagation).

<br>

## [Two-Layer Perceptron Architecture]()

A two-layer perceptron (MLP with one hidden layer and one output layer) can approximate any function, linear or not (Cybenko, 1989). 

- [**Layer 1 (Hidden/Intermediate):**]() Each neuron contributes lines (hyperplanes) to form surfaces in input space, "linearizing" the features.

- [**Layer 2 (Output):**]() Neurons combine these lines to form convex regions, enabling complex decision boundaries.


<br>

[**Number of Neurons:**]()

- The generalization capacity of the network increases with the number of neurons.

- Empirically, 3–5 neurons per layer strike a good balance between modeling power and computational cost.

<br>

[**Layer Types:**]()

- [**Input Layer:**]() Receives input patterns.

- [**Hidden Layer(s):**]() Main processing; feature extraction.

- [**Output Layer:**]() Produces the final result.

<br>

## [Main Concepts and Key Formulas]()

<br>

- [**Neuron Activation:**]()

<br>  
  
  $$
  \Huge
  a = \sum_{i=1}^{p} w_i X_i
  $$

<br><br>
  
  
#


- [**Output:**]()

<br>
  
$$
\Huge y = f(a)
$$

### <p align="center"> [where]() $\Huge f$ is the activation function (e.g., sigmoid, tanh)
  

<br><br>
  
  
#

- [**Error Calculation:**]()

<br>

  $$
  \Huge
  e_k(n) = d_k(n) - y_k(n)$
  $$ 

<br><br>
  
#

- [**Cost Function (Mean Squared Error):**]()

<br>
  
$$
\Huge
E(n) = \frac{1}{2} e_k^2(n
$$

<br><br>
  
#  
  
- [**Weight Update (Gradient Descent):**]()

<br>

$$
\Huge
w_{kj}(n+1) = w_{kj}(n) + \eta \frac{\partial E(n)}{\partial w_{kj}}
$$

<!--

$\Huge w_{kj}(n+1) = w_{kj}(n) + \eta \frac{\partial E(n)}{\partial w_{kj}}$
  
$w_{kj}(n+1) = w_{kj}(n) + \eta \frac{\partial E(n)}{\partial w_{kj}}$
-->

<br><br>
  
#

- [**Backpropagation for Output Layer:**]()

<br>

$$
\Huge
\delta^{(2)}(t) = (d(t) - y(t)) \cdot f'^{(2)}(u)
$$
 
<br><br>
  
#


- [**Backpropagation for Hidden Layer:**]()

<br>

$$
\Huge
delta_j^(1)(t) = ( sum_k [ delta_k^(2) * w_kj^(2) ] ) * f'^(1)( u_j^(1))
$$


<br><br>
  

## [Training](): Two-Phase Process

<br>

### [1](). Forward Phase*

- Initialize learning rate $\eta$ and weight matrix $w$ with random values.  

- Present input to the first layer.  

- Each neuron in layer $i$ computes its output, which is passed to the next layer.  

- The final output is compared to the desired output.  

- The error for each output neuron is calculated.


<br>

### - [**Example Calculation:**]()

### Forward Computation Example

### - [For input values]():

<br>

- $\Large \( X_0 = 1 \)$

<br>

- $\Large \( X_1 = 0.43 \)$

<br>

- $\Large \( X_2 = 0.78 \)$

<br>


### - [And example weights]():

<br>

- $\Large \( w^{(1)}_{00} = 0.45 \)$

<br>

- $\Large \( w^{(1)}_{01} = 0.89 \)$
  
<br>

- etc...


<br><br>

## [Compute the activations and outputs for each layer using an activation function (e.g.,** `tanh`**)]():

<br>

- ### [**Compute pre-activation (input to each hidden neuron)**]():

<br>

$$
u_j^{(1)} = \sum_i X_i \cdot w_{ji}^{(1)}
$$

<br>

- **Compute activation (output from each hidden neuron):**

  $y^{(1)}_j = \tanh(u^{(1)}_j)$

- Compute output layer pre-activation:  
  $u^{(2)} = \sum_j y^{(1)}_j w^{(2)}_j$
  
- Output of network:  
  $y^{(2)} = \tanh(u^{(2)})$

- Calculate error:  
  $e = d - y^{(2)}$  
  $E = \frac{1}{2} e^2$

<br>

### [2](). Backward Phase (Backpropagation)

- Start from the output layer.  

- Each node adjusts its weight to reduce its error.  

- For hidden layers, the error is determined by the weighted errors of the next layer (chain rule).

- Output layer weight update:  

  $w^{(2)}(t+1) = w^{(2)}(t) + \eta \delta^{(2)} y^{(1)}(t)$  
  
  where $\delta^{(2)}(t) = (d(t) - y(t)) \cdot f'^{(2)}(u)$

- Hidden layer delta:

 ```latex
  $\delta^{(1)}_j(t) = \left( \sum_k \delta^{(2)}_k w^{(2)}_{kj} \right) \cdot f'^{(1)}(u_j)$
```

<br>

## Example: Training a Two-Layer Perceptron

[1](). **Initialize all weights randomly.**
[2](). **Present an input vector $X$.**
[3.]() **Compute outputs for the first (hidden) layer:**
   
$u_j^{(1)} = \sum_i X_i w_{ji}^{(1)}$

$y_j^{(1)} = \tanh(u_j^{(1)})$


   
[4](). **Compute output for the second (output) layer:**

   $u^{(2)} = \sum_j y^{(1)}_j \cdot w^{(2)}_j$

   $y^{(2)} = \tanh(u^{(2)})$


[5](). **Calculate error:**

   $e = d - y^{(2)}$

   $E = \frac{1}{2} e^2$


[6](). **Backward phase:**

   - Compute $\delta^{(2)}$ and update output weights.
   - Compute $\delta^{(1)}$ for each hidden neuron and update hidden weights.

  <br>

## [Why Two Layers and 3–5 Neurons per Layer?]()

- **Theoretical Power:** Two-layer MLPs can approximate any continuous function (universal approximation theorem).
- **Practical Simplicity:** Most real-world problems rarely require more than two layers.
- **Cost-Benefit:** 3–5 neurons per layer often provide sufficient capacity for generalization without excessive computational cost.


<br>

## [Local Maximum (Local Maxima)]()

In gradient descent training, the algorithm updates weights to reduce error by following the gradient of the cost function. However, the cost function may have multiple local maxima or minima. 

- [**Local Maximum:**]() A point where the cost function has a peak relative to nearby points but is not the absolute highest point globally.

- Gradient descent can [get "stuck"]() in local maxima or minima, preventing the network from reaching the best possible solution.

- Techniques such as [random restarts](), [momentum](), or [advanced optimization algorithms]() help mitigate this problem.


<br>

## [Usage]()

Artificial Neural Networks, especially perceptrons and MLPs, are widely used in various domains due to their adaptability and ability to model complex nonlinear relationships.

<br>

### [Strengths]()

- Ability to learn from examples and generalize to unseen data.

- Fault tolerance and robustness to noisy inputs.

- Flexibility to model complex, nonlinear functions.

- Parallel processing capability.

<br>

### [Weaknesses]()

- Training can be computationally expensive, especially for large networks.

- Susceptible to getting stuck in local minima or maxima.

- Requires careful tuning of hyperparameters (learning rate, number of neurons, layers).

Lack of interpretability compared to simpler models.


<br>

## [Additional Relevant Points]()

### Learning Rate (η) Importance

The learning rate $\eta$ controls the step size during weight updates:

- If $\eta$ is too large, the training may overshoot minima and fail to converge.

- If $\eta$ is too small, training will be very slow and may get stuck in local minima.

- Adaptive learning rate methods (e.g., learning rate decay, Adam optimizer) can improve convergence.


<br>

### [Activation Functions]()

While the document mentions sigmoid and tanh, it is useful to note:

- [**ReLU (Rectified Linear Unit):**]()  
  Widely used in modern neural networks for faster convergence and to mitigate vanishing gradient problems.
  
- [**Softmax:**]()  
  Commonly used in output layers for multi-class classification problems.



### [Overfitting and Regularization]()

- Neural networks with too many parameters can overfit training data, performing poorly on unseen data.

- Techniques such as early stopping, dropout, and L2 regularization help improve generalization.



### [Batch vs. Online Learning]()

- The document discusses iterative weight updates per sample (online/stochastic gradient descent).

- In practice, **batch** or **mini-batch** gradient descent is often used for computational efficiency and stability.



### [Practical Considerations]()

- Data preprocessing (normalization, encoding) is crucial for effective training.

- Initialization of weights affects convergence speed and final performance.

- Monitoring training with validation sets helps detect overfitting.


<br>

 # [Algorithms Used to Train Machine Learning Models]()

## 1.[Gradient Descent]()

Gradient Descent is a mathematical optimization method primarily used for minimizing differentiable multivariate functions. It is a first-order iterative algorithm that adjusts model parameters to find the minimum value of a function, typically representing an error or cost to minimize.

The way gradient descent works can be explained as follows: Imagine standing on top of a hill wanting to reach the lowest point in a valley. In algorithm terms, you start with initial parameter values and calculate the slope (gradient) of the cost function with respect to these parameters. This slope shows the steepest ascent direction. To minimize the function, you take a step in the opposite direction, "descending the slope" toward the lowest point.

These steps are repeated iteratively, adjusting the model parameters opposite to the gradient direction until the algorithm converges to the minimum. The step size is controlled by a learning rate that defines how big the adjustments are at each iteration.


<br>

### 1 - [Gradient Descent (**Batch**)]()

Gradient Descent is an iterative algorithm to minimize a cost function by adjusting parameters opposite to the gradient direction. Batch Gradient Descent calculates the gradient using the entire dataset each step, resulting in stable but sometimes slow parameter updates.


<br>

### 2 [Stochastic Gradient Descent (SGD)]()

Stochastic Gradient Descent updates parameters based on a single random sample per iteration. This yields noisier but faster updates, suitable for large datasets and deep learning models.


<br>

### 3. [Elastic Net Regularization]()

Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties to improve the performance of linear regression models, particularly when many variables are correlated. It helps prevent overfitting and performs automatic feature selection, making it a powerful tool for machine learning modeling.



### 4. [Mini-batch Gradient Descent]()

Mini-batch Gradient Descent is a compromise between batch and stochastic descent. It updates parameters using small random batches, accelerating convergence with reduced noise.

<br>


### 5. [Adam (Adaptive Moment Estimation)]() 

An algorithm combining momentum and adaptive learning rates to improve convergence and training efficiency, especially in deep neural networks.


<br>

### 6. [RMSProp]() 

Adapts the learning rate for each parameter, useful to accelerate training and avoid oscillations.



<br><br>




# II - [Artificial Neural Networks (ANN) - Comprehensive Theory, Use Cases, and Python Code Guide]()

<br>

### Required Libraries Installation

Before running any code, install the necessary Python libraries appropriate for your operating system and environment:

<br><br>

#### [Cell 1]() - Installation Commands

<br>


```python
# macOS Terminal or Jupyter Notebook (IPython):

%pip install numpy matplotlib tensorflow scikit-learn tensorflow-datasets
```

<br>

```python
# Windows Command Prompt or PowerShell:

pip install numpy matplotlib tensorflow scikit-learn tensorflow-datasets
```

<br>


```python
# Linux Terminal or Jupyter Notebook (IPython):

%pip install numpy matplotlib tensorflow scikit-learn tensorflow-datasets
```

<br>


> This will install:
> - **numpy**: Numerical computations
> - **matplotlib**: Plotting and visualization
> - **tensorflow**: Deep learning framework
> - **scikit-learn**: Machine learning utilities
> - **tensorflow-datasets**: Loading datasets like MNIST easily


<br>


## 0. [Understanding Tensors and Loading MNIST Dataset]()

### 0.1 [What is a Tensor]()?

### - [**Concept:**]()  

A tensor generalizes scalars (0-D), vectors (1-D), and matrices (2-D) to n-dimensional arrays. Data in neural networks (inputs, weights, activations) are represented as tensors.  
Understanding tensors is essential for deep learning frameworks.

### - [**Use Case]()  

Manages multi-dimensional data like images (3D tensors with height, width, channels) or batches of images (4D tensors).


### - [**Code:**]()

<br>

```python
import tensorflow as tf

# 0-D scalar

scalar = tf.constant(42)
print("Scalar:", scalar, "Shape:", scalar.shape)

# 1-D vector

vector = tf.constant()
print("Vector:", vector, "Shape:", vector.shape)

# 2-D matrix

matrix = tf.constant([, ])
print("Matrix:\n", matrix.numpy())
print("Shape:", matrix.shape)

# 3-D tensor (example: color image 2x2 pixels with 3 color channels)

tensor_3d = tf.constant([[, ],
[, ]])
print("3D tensor:\n", tensor_3d.numpy())
print("Shape:", tensor_3d.shape)
```

<br><br>


### 0.2 [Loading MNIST Dataset from TensorFlow Datasets]()

### - [**Concept:**]()  

MNIST dataset can be streamed using TensorFlow Datasets, cached automatically without manual download management.


### - [**Use Case:**]()  

Practice and benchmark image classification models.


### - [**Code:**]()

<br>

```
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

ds_train = tfds.load('mnist', split='train', shuffle_files=True, as_supervised=True)
ds_test = tfds.load('mnist', split='test', as_supervised=True)

def normalize_img(image, label):
image = tf.cast(image, tf.float32) / 255.0
return image, label

ds_train = ds_train.map(normalize_img).shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img).batch(32).prefetch(tf.data.AUTOTUNE)

for image, label in ds_train.take(1):
plt.imshow(tf.squeeze(image), cmap='gray')
plt.title(f"Label: {label.numpy()}")
plt.axis('off')
plt.show()
```

<br><br>


## 1. [Artificial Neuron Model]()

### - [**Concept:**]()  

Computes weighted sum of inputs followed by a nonlinear activation function (e.g., sigmoid).


### - [**Use Case:**]()  

Fundamental computation unit for classification.


### - [**Code:**]() -  with MNIST-like input shape simplified to vector

<br>

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
return 1 / (1 + np.exp(-x))

# Simulating a single flattened MNIST image input (784 pixels normalized)

X = np.random.rand(784)  \# Normally we'd flatten an image; here random for example
w = np.random.rand(784)  \# weights vector

a = np.dot(w, X)
y = sigmoid(a)

print("Activation:", a)
print("Output:", y)

x_vals = np.linspace(-10, 10, 100)
plt.plot(x_vals, sigmoid(x_vals))
plt.title("Sigmoid Activation")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()
```

<br><br>


## 2. [Gradient Descent (Batch Gradient Descent)]()

### -  [**Concept:**]()  

Updates weights by calculating gradients over the entire training dataset.


### - [**Use Case:**]()  

Stable, but computationally expensive for large datasets.


### - [**Code:**]()

<br>


```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data generation for demonstration

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  \# add bias term

learning_rate = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)  \# initial weights

for iteration in range(n_iterations):
gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
theta = theta - learning_rate * gradients

print("Theta (Batch GD):", theta)

plt.plot(X, y, "b.")
X_new = np.array([, ])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta)
plt.plot(X_new, y_predict, "r-")
plt.title("Batch Gradient Descent")
plt.show()
```

<br><br>


## 3. [Stochastic Gradient Descent (SGD)]()

### - [**Concept:**]()  

Updates weights using gradients from one training sample at a time, adding noise but enabling faster updates.

### - [**Use Case:**]()  

Useful for large datasets or online learning.

### - [**Code:**]()

<br>

```python
theta = np.random.randn(2,1)
n_epochs = 50
t = 0
m = len(X_b)

for epoch in range(n_epochs):
for i in range(m):
random_index = np.random.randint(m)
xi = X_b[random_index:random_index+1]
yi = y[random_index:random_index+1]
gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
eta = 0.1 / (1 + t * 0.01)  \# decaying learning rate
theta = theta - eta * gradients
t += 1

print("Theta (SGD):", theta)
```

<br><br>


## 4. [Mini-batch Gradient Descent]()

### - [**Concept:**]()  

Updates weights on small subsets (mini-batches), balancing stability and speed.

### - [**Use Case:**]() 

Common in deep learning training.

### - [**Code:**]()

<br>

```python
theta = np.random.randn(2,1)
n_iterations = 50
batch_size = 20
m = 100

for iteration in range(n_iterations):
indices = np.random.permutation(m)
for start_idx in range(0, m, batch_size):
end_idx = start_idx + batch_size
X_batch = X_b[indices[start_idx:end_idx]]
y_batch = y[indices[start_idx:end_idx]]
gradients = 2/len(X_batch) * X_batch.T.dot(X_batch.dot(theta) - y_batch)
theta = theta - learning_rate * gradients

print("Theta (Mini-batch):", theta)
```

<br><br>


## 5. [Elastic Net Regularization with Scikit-learn]()

### - [**Concept:**]()  

Combines L1 and L2 regularization to prevent overfitting.

### - [**Use Case:**]() 

Regression with regularization on MNIST features or other datasets.

### - [**Code:**]()

<br>

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example with synthetic or MNIST flattened feature data

X_train, X_test, y_train, y_test = train_test_split(X_b[:,1].reshape(-1,1), y, test_size=0.2, random_state=42)

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=1000)
elastic_net.fit(X_train, y_train.ravel())
y_pred = elastic_net.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"ElasticNet Coefs: {elastic_net.coef_}")
print(f"Intercept: {elastic_net.intercept_}")
print(f"MSE: {mse}")
```

<br><br>



## 6. [Multilayer Perceptron (MLP) Forward Pass]()

### - [**Use Case:**]()  

General-purpose nonlinear function approximation with hidden layers.

### - [**Code:**]()

<br>

```python
import numpy as np

X = np.random.rand(785)  \# example input with bias
w_hidden = np.random.rand(2, 785)  \# two neurons in hidden layer

u_hidden = np.dot(w_hidden, X)
y_hidden = np.tanh(u_hidden)

w_output = np.random.rand(2)
u_output = np.dot(w_output, y_hidden)
y_output = np.tanh(u_output)

print("Hidden outputs:", y_hidden)
print("Network output:", y_output)
```

<br><br>


## 7. [Backpropagation and Weight Updates]()

### - [**Use Case:**]()

Train neural networks by propagating error gradients backward.

### - [**Code:**]()

<br>

```python
desired = 0.5
error = desired - y_output
E = 0.5 * error**2

def tanh_derivative(x):
return 1 - np.tanh(x)**2

delta_output = error * tanh_derivative(u_output)
delta_hidden = delta_output * w_output * tanh_derivative(u_hidden)

learning_rate = 0.1

w_output += learning_rate * delta_output * y_hidden
w_hidden += learning_rate * np.outer(delta_hidden, X)

print("Updated weights")
print("Output weights:", w_output)
print("Hidden weights:", w_hidden)
print("Loss:", E)
```

<br><br>


## 8. [Momentum Optimization]()

### - [**Use Case:**]()

Speeds convergence and helps avoid local minima during training.

### - [**Code:**]()

<br>

```python
import numpy as np

grad = np.array([0.1, -0.2, 0.05])
learning_rate = 0.1
momentum = 0.9
velocity = np.zeros_like(grad)
weights = np.array([0.5, -0.3, 0.8])

velocity = momentum * velocity - learning_rate * grad
weights += velocity

print("Weights updated:", weights)
```

<br><br>


## 9. [Activation Functions: ReLU Example]()

### - [**Use Case:**]()

Effective activation to accelerate deep neural network training.

### - [**Code:**]()

<br>

```python
import numpy as np

def relu(x):
return np.maximum(0, x)

inputs = np.array([-2, -1, 0, 1, 2])
outputs = relu(inputs)

print("ReLU outputs:", outputs)
```

<br><br>


## 10. [Regularization & Dropout Example]()

### - [**Use Case:**]()

Combat overfitting by randomly disabling neurons during training.

### - [**Code:**]()

<br>

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
Dense(64, activation='relu', input_shape=(20,)),
Dropout(0.5),
Dense(1, activation='sigmoid')
])

model.summary()
```

<br><br>

## 11. [Batch vs Mini-batch vs Online Learning]()

### - [**Use Case:**]()

Trade-offs in training efficiency and noise introduced.

### - [**Code:**]()

<br>

```python
import numpy as np

batch_size = 32
dataset_size = 1000
indices = np.arange(dataset_size)

for epoch in range(5):
np.random.shuffle(indices)
for start in range(0, dataset_size, batch_size):
end = start + batch_size
batch_indices = indices[start:end]
\# Perform training step on batch_data
```

<br><br>


## 12. [Data Preprocessing: Feature Normalization]()

### - [**Use Case:**]()

Normalize features to accelerate and stabilize training.

### - [**Code:**]()

<br>

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.random.rand(100, 5) * 10

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

print("Means after scaling:", np.mean(X_normalized, axis=0))
print("Stds after scaling:", np.std(X_normalized, axis=0))
```

<br><br>


## 13. [Adam Optimizer Example with MNIST]()

### - [**Use Case:**]()

Efficient training with adaptive gradients.

### - [**Code:**]()

<br>

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
Flatten(input_shape=(28, 28)),
Dense(128, activation='relu'),
Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
```

<br><br>


## 14. [RMSProp Optimizer Example with MNIST]()

### - [**Use Case:**]()

Alternative adaptive optimizer for neural networks.

### - [**Code:**]()

<br>

```python
model_rms = Sequential([
Flatten(input_shape=(28, 28)),
Dense(128, activation='relu'),
Dense(10, activation='softmax')
])

model_rms.compile(optimizer='rmsprop',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

model_rms.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)
test_loss, test_acc = model_rms.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy with RMSProp: {test_acc:.4f}")
```

<br><br>



## 15. [Demonstrating Accuracy with MNIST Dataset]()

To demonstrate the accuracy of your trained model on the MNIST dataset using TensorFlow, you should use the model's `evaluate()` method. This method returns both the loss and the evaluation metrics such as accuracy.

### - [**Code Example:**]()

<br>

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model specifying optimizer, loss, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=2)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Print the accuracy as a percentage
print(f"Test accuracy: {test_acc * 100:.2f}%")
```

<br><br>

### [Explanation]()

- The `evaluate()` function runs the model on the test data and reports the loss and accuracy.

- The accuracy is printed as a percentage for easier interpretation.

- This approach is standard for classification tasks and directly shows how well your model performs on unseen data


<br><br>


## 16. [Overfitting and Early Stopping]()

### - [Concept:**]()

Overfitting means a model fits training data too closely including noise, harming generalization to new data.

### - [**Use Case:**]()

Early Stopping monitors validation loss and stops training when improvements cease, mitigating overfitting. This improves generalization in MNIST and other datasets.

<br>

### ⚠️ [Important:]() MNIST Loading and Normalization (Place This at the Start of Your Notebook/Script)

Before running the training code below, ensure you have loaded and normalized the MNIST dataset so that the variables `x_train` and `y_train` exist and are ready for use.


<br>

```python
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to
x_train = x_train / 255.0
x_test = x_test / 255.0
```

<br><br>

### [Early Stopping Model Training Code (Sendoultimo Item):]()

<br>

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=50,
          validation_split=0.1,
          callbacks=[early_stopping],
          verbose=2)
```

<br><br>

### [Summary of Placement:]()

- [**MNIST loading and normalization block**]() should be inserted at the **very start** of your notebook or script (immediately after imports).

- The [**Early Stopping model training code block**]() should be placed later in the **training section**, near the end of your model building and training workflow, but before evaluation.

<br>

This arrangement ensures that your model training example with early stopping is fully functional using the MNIST dataset, and provides a clear, logical flow for this script.

<br><br>



































<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


## [References]()

- Content derived from [Decreasing-Gradient.pdf](./Decreasing-Gradient.pdf).
- Classic works by McCulloch & Pitts, Hebb, Rosenblatt, Hopfield, Rumelhart, Hinton & Williams, and Cybenko.
- NVIDEA Building a Brain Course
- Neuralearn Courses

<br><br>


## See alsso our Project:
### [Predictive, PI, and Gradient Descent Control in TAB Converters for Electric Vehicles](https://github.com/Mindful-AI-Assistants/brains-made-of-code-ml-gd-sgd/tree/c23e6b6b7dc47832b7ab32aedfd2f3815e5cdbd3/Projec-APPLICATION%20OF%20MPC%20CONTROLS%20WITH%20DESCENDING%20GRADIENT%20AND%20PI%20IN%20A%20TAB%20CONVERTER%20USED%20IN%20ELECTRIC%20VEHICLE%20POWERTRAINS) 
#### 🚛 (Under Construtction)


<br>


## ✌️ Meet the Crew — Under Jah’s Vibes! 🟥🟨🟩  

<br><br>


<!--
π is the most famous number in the world. It is a irrational number, meaning it cannot be written exactly as a ratio of two integers. However, one can still approximate it by a ratio!
-->

https://github.com/user-attachments/assets/f8ad6d7a-6d85-4f2c-b9cc-ce8230ba3b9b

<br><br>



- ࣪ 𖤐 [Andson Ribeiro](https://github.com/andsonandreribeiro09) 
- ࣪ 𖤐 [Fabiana 🚀 Campanari](https://github.com/FabianaCampanari)  
- ࣪ 𖤐 [Leonardo XF](https://github.com/LeonardoXF)
- ࣪ 𖤐 [Pedro 🛰️ Vyctor Almeida](https://github.com/ppvyctor) 

<br>

➣ United by [Vision]()

➢ Guided by [Jah]() 

➣ Strength in [Unity]() ≽༏≼⊹  


<br>

## Reference

- Content derived from [Decreasing-Gradient.pdf](./Decreasing-Gradient.pdf).
  
- [Application of MPC controls with descending gradient and PI in a TAB converter used in electric vehicle powertrains](https://github.com/Mindful-AI-Assistants/brains-made-of-code-ml-gd-sgd/blob/c64e25ed0edf8aa7eed48f0627299f12db45fb2e/project_Predictive%2C%20PI%2C%20and%20Gradient%20Descent%20Control%20in%20TAB%20Converters%20for%20Electric%20Vehicles/Application%20of%20MPC%20controls%20with%20descending%20gradient%20and%20pi%20in%20a%20tab%20converter%20used%20in%20electric%20vehicle%20powertrains.pdf) by Atílio Caliari de Lima,PHD.


<br><br>

##  Feel Free to [Reach Out:]()

### 💌 [Email Me](mailto:fabicampanari@proton.me)

<br>


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />


<br><br>

<p align="center">  ────────────── ⊹🔭๋ ──────────────

<!--
<p align="center">  ────────────── 🛸๋*ੈ✩* 🔭*ੈ₊ ──────────────
-->

<br>

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>
  

  
#
 
##### <p align="center">Copyright 2025 Mindful-AI-Assistants. Code released under the  [MIT license.]( https://github.com/Mindful-AI-Assistants/.github/blob/ad6948fdec771e022d49cd96f99024fcc7f1106a/LICENSE)





