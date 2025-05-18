
<br>

## 🧠  [Brains Made of Code](): Machine Learning Regression with Gradient Descent and Stochastic Optimization

<br>


![Image](https://github.com/user-attachments/assets/c1d1a6ae-cf3d-4a71-871f-461928012002)

<br>

This project demonstrates how to train a Linear Regression model using both Batch Gradient Descent (GD) and Stochastic Gradient Descent (SGD). The implementation includes Python code, dataset, and detailed visualizations to illustrate convergence behavior, performance comparison, and optimization dynamics.

Ideal for beginners and intermediate learners looking to understand the foundations of machine learning optimization algorithms.


<br>


# Artificial Neural Networks – Gradient Descent

This repository provides a comprehensive explanation of Artificial Neural Networks (ANNs), focusing on the perceptron and multilayer perceptron (MLP) architectures, and the Gradient Descent algorithm for training. The content is based on the [Decreasing-Gradient.pdf](./Decreasing-Gradient.pdf) document.

<br>

## Motivation

The human brain processes information in a highly complex, nonlinear, and parallel way, which is fundamentally different from conventional digital computers. For example, tasks such as visual recognition (e.g., recognizing a familiar face in an unfamiliar scene) are performed by the brain in milliseconds, while much simpler tasks can take a conventional computer days to complete.

At birth, a child's brain has a large structure and the ability to develop its own rules through experience. ANNs are computational machines designed to model or simulate the way the brain performs specific tasks or functions of interest.

<br>

## Historical Context

- **McCulloch & Pitts (1943):** Introduced the first neural network models.
- **Hebb (1949):** Developed the basic model of self-organization.
- **Rosenblatt (1958):** Introduced the perceptron, a supervised learning model.
- **Hopfield (1982), Rumelhart, Hinton & Williams:** Revived the field with symmetric networks for optimization and the backpropagation method.
  
<br>

## Artificial Neuron Model

### Each artificial neuron receives input signals $X_1, X_2, ..., X_p$ (binary or real values), each multiplied by a weight $w_1, w_2, ..., w_p$ (real values). The neuron computes a weighted sum (activity level):

$$
a = w_1 X_1 + w_2 X_2 + \cdots + w_p X_p
$$


```latex
\a = w_1 X_1 + w_2 X_2 + \cdots + w_p X_p\
```

<br>

### The output $y$ is determined by an activation function, such as:

$$
y =
\begin{cases}
1, & \text{if } a \geq t \\
0, & \text{if } a < t
\end{cases}
$$

```latex
y =
\begin{cases}
1, & \text{if } a \geq t \\
0, & \text{if } a < t
\end{cases}
```
  
<br>


## Key Benefits of ANNs

- Adaptability through learning
- Ability to operate with partial knowledge
- Fault tolerance
- Generalization
- Contextual information processing
- Input-output mapping

## Application Areas

- Pattern classification
- Clustering/categorization
- Function approximation
- Prediction
- Optimization
- Content-addressable memory
- Control systems

  <br>

## Learning Process

ANNs operate in two main phases:

1. **Training Phase:** The network learns by adjusting its free parameters (weights) to perform a specific function.
2. **Application Phase:** The trained network is used for its intended purpose (e.g., pattern or image classification).

## The learning process involves:

1. Stimulation by the environment (input).
2. Modification of free parameters (weights) as a result.
3. The network responds differently due to internal changes.

Learning is governed by a set of pre-established rules (learning algorithm) and a learning paradigm (model).

<br>

## Error Correction Learning

The output of neuron $k$ at iteration $n$ is $y_k(n)$, and the desired response is $d_k(n)$. The error signal is:

$$
e_k(n) = d_k(n) - y_k(n)
$$

The goal is to minimize the cost function (performance index):

$$
E(n) = \frac{1}{2} e_k^2(n)
$$

Weights are updated as:

$$
w_{kj}(n+1) = w_{kj}(n) + \Delta w_{kj}(n)
$$

<br>

# The Perceptron

The perceptron, proposed by Rosenblatt (1958), is the simplest type of ANN. It uses supervised learning and error correction to adjust the weight vector. For a perceptron with two inputs and a bias:

- The bias allows the threshold value in the activation function to be set, and is updated like any other weight.

<br>

## Nonlinearities and Activation Functions

- Nonlinearities are inherent in most real-world problems.
- Incorporated through nonlinear activation functions (e.g., sigmoid, tanh) and multiple layers.
- MLPs use sigmoid functions in hidden layers and linear functions in the output layer.

<br>

## MLP (MultiLayer Perceptron)

- Composed of neurons with nonlinear activation functions in intermediate (hidden) layers.
- Only the output layer receives a desired output during training.
- The error for hidden layers is estimated by the effect they cause on the output error (backpropagation).

<br>

# Two-Layer Perceptron Architecture

A two-layer perceptron (MLP with one hidden layer and one output layer) can approximate any function, linear or not (Cybenko, 1989). 

- **Layer 1 (Hidden/Intermediate):** Each neuron contributes lines (hyperplanes) to form surfaces in input space, "linearizing" the features.
- **Layer 2 (Output):** Neurons combine these lines to form convex regions, enabling complex decision boundaries.

**Number of Neurons:**  
- The generalization capacity of the network increases with the number of neurons.
- Empirically, 3–5 neurons per layer strike a good balance between modeling power and computational cost.

**Layer Types:**
- **Input Layer:** Receives input patterns.
- **Hidden Layer(s):** Main processing; feature extraction.
- **Output Layer:** Produces the final result.

<br>

## Main Concepts and Key Formulas

- **Neuron Activation:**  
  $a = \sum_{i=1}^{p} w_i X_i$
- **Output:**  
  $y = f(a)$, where $f$ is the activation function (e.g., sigmoid, tanh)
- **Error Calculation:**  
  $e_k(n) = d_k(n) - y_k(n)$
- **Cost Function (Mean Squared Error):**  
  $E(n) = \frac{1}{2} e_k^2(n)$
- **Weight Update (Gradient Descent):**  
  $w_{kj}(n+1) = w_{kj}(n) + \eta \frac{\partial E(n)}{\partial w_{kj}}$
- **Backpropagation for Output Layer:**  
  $\delta^{(2)}(t) = (d(t) - y(t)) \cdot f'^{(2)}(u)$
- **Backpropagation for Hidden Layer:**  
  $\delta^{(1)}_j(t) = \left( \sum_k \delta^{(2)}_k w_{kj}^{(2)} \right) \cdot f'^{(1)}(u_j)$

<br>

## Training: Two-Phase Process


