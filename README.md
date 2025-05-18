
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

ach artificial neuron receives input signals $X_1, X_2, ..., X_p$ (binary or real values), each multiplied by a weight $w_1, w_2, ..., w_p$ (real values). The neuron computes a weighted sum (activity level):

$$
a = w_1 X_1 + w_2 X_2 + \cdots + w_p X_p
$$

The output $y$ is determined by an activation function, such as:

$$
y =
\begin{cases}
1, & \text{if } a \geq t \\
0, & \text{if } a < t
\end{cases}
$$

  
<br>
