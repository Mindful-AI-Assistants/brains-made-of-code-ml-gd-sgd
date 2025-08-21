
<br>

#  <p align="center"> 🧠  [Brain]() Made of Code, [Created]() with [Heart]() ❤︎
### <p align="center"> Machine Learning Regression with Gradient Descent and Stochastic Optimization

<br><br>


#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-Mindful%20AI%20%20Assistants-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)

<br><br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/20470843-6f04-4e20-81d5-7aa918ca9a2b" width="450"/>
</p>


<br><br>

This project demonstrates how to train a Linear Regression model using both Batch Gradient Descent (GD) and Stochastic Gradient Descent (SGD). The implementation includes Python code, dataset, and detailed visualizations to illustrate convergence behavior, performance comparison, and optimization dynamics.

Also presents key optimization algorithms for machine learning with explanations followed by Python code split into individual cells. Each algorithm’s explanation precedes its corresponding step-by-step code cells, ready for direct copy-pasting into Jupyter Notebook or Google Colab.

Ideal for beginners and intermediate learners looking to understand the foundations of machine learning optimization algorithms.

<br><br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c1d1a6ae-cf3d-4a71-871f-461928012002" />

<br><br>



      
# [Artificial Neural Networks – Gradient Descent]()

This repository provides a comprehensive explanation of Artificial Neural Networks (ANNs), focusing on the perceptron and multilayer perceptron (MLP) architectures, and the Gradient Descent algorithm for training. The content is based on the [Decreasing-Gradient.pdf](./Decreasing-Gradient.pdf) document.

<br>

## [Motivation]()

The human brain processes information in a highly complex, nonlinear, and parallel way, which is fundamentally different from conventional digital computers. For example, tasks such as visual recognition (e.g., recognizing a familiar face in an unfamiliar scene) are performed by the brain in milliseconds, while much simpler tasks can take a conventional computer days to complete.

At birth, a child's brain has a large structure and the ability to develop its own rules through experience. ANNs are computational machines designed to model or simulate the way the brain performs specific tasks or functions of interest.

<br>

## [Historical Context]()

- **McCulloch & Pitts (1943):** Introduced the first neural network models.
- **Hebb (1949):** Developed the basic model of self-organization.
- **Rosenblatt (1958):** Introduced the perceptron, a supervised learning model.
- **Hopfield (1982), Rumelhart, Hinton & Williams:** Revived the field with symmetric networks for optimization and the backpropagation method.
  
<br>

## [Artificial Neuron Mode]()

### Each artificial neuron receives input signals $X_1, X_2, ..., X_p$ (binary or real values), each multiplied by a weight $w_1, w_2, ..., w_p$ (real values). The neuron computes a weighted sum (activity level):

<br>

$$
\Huge
a = w_1 X_1 + w_2 X_2 + \cdots + w_p X_p
$$

<br>

```latex
\a = w_1 X_1 + w_2 X_2 + \cdots + w_p X_p\
```

<br>

### [The output $y$ is determined by an activation function, such as:]()

<br>

$$
\Huge
y =
\begin{cases}
1, & \text{if } a \geq t \\
0, & \text{if } a < t
\end{cases}
$$

<br>

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

<br>

### [The goal is to minimize the cost function (performance index):]()

<br>

$$
\Huge
E(n) = \frac{1}{2} e_k^2(n)
$$

<br>

### [Weights are updated as:]()

<br>

$$
\Huge
w_{kj}(n+1) = w_{kj}(n) + \Delta w_{kj}(n)
$$

<br>

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

- **Input Layer:** Receives input patterns.
- **Hidden Layer(s):** Main processing; feature extraction.
- **Output Layer:** Produces the final result.

<br>

## [Main Concepts and Key Formulas]()

<br>

- [**Neuron Activation:**]()

<br>  
  
  $$
  \Huge
  a = \sum_{i=1}^{p} w_i X_i
  $$

<br>
  

- [**Output:**]()

  <br>
  
  $\Huge y = f(a)$, where $f$ is the activation function (e.g., sigmoid, tanh)
  

<br>
  

- [**Error Calculation:**]()
  
  $e_k(n) = d_k(n) - y_k(n)$
  

<br>
  


- [**Cost Function (Mean Squared Error):**]()

<br>
  

  $E(n) = \frac{1}{2} e_k^2(n)$

<br>
  
  
- [**Weight Update (Gradient Descent):**]()


  
  $w_{kj}(n+1) = w_{kj}(n) + \eta \frac{\partial E(n)}{\partial w_{kj}}$
  

- [**Backpropagation for Output Layer:**]()

  $\delta^{(2)}(t) = (d(t) - y(t)) \cdot f'^{(2)}(u)$


[- **Backpropagation for Hidden Layer:**]()


```latex
delta_j^(1)(t) = ( sum_k [ delta_k^(2) * w_kj^(2) ] ) * f'^(1)( u_j^(1))
```

<br>


## [Training](): Two-Phase Process

### [1](). Forward Phase

- Initialize learning rate $\eta$ and weight matrix $w$ with random values.  
- Present input to the first layer.  
- Each neuron in layer $i$ computes its output, which is passed to the next layer.  
- The final output is compared to the desired output.  
- The error for each output neuron is calculated.


### [**Example Calculation:**]()

### Forward Computation Example

For input values:

- \( X_0 = 1 \)
- \( X_1 = 0.43 \)
- \( X_2 = 0.78 \)

And example weights:

- \( w^{(1)}_{00} = 0.45 \)
- \( w^{(1)}_{01} = 0.89 \)
- etc.


**Compute the activations and outputs for each layer using an activation function (e.g.,** `tanh`**):**

<p>Compute pre-activation (input to each hidden neuron):</p>
<p>$$
u_j^{(1)} = \sum_i X_i \cdot w_{ji}^{(1)}
$$</p>

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

 # [Algorithms]()

## 1.[Gradient Descent]()

Gradient Descent is a mathematical optimization method primarily used for minimizing differentiable multivariate functions. It is a first-order iterative algorithm that adjusts model parameters to find the minimum value of a function, typically representing an error or cost to minimize.

The way gradient descent works can be explained as follows: Imagine standing on top of a hill wanting to reach the lowest point in a valley. In algorithm terms, you start with initial parameter values and calculate the slope (gradient) of the cost function with respect to these parameters. This slope shows the steepest ascent direction. To minimize the function, you take a step in the opposite direction, "descending the slope" toward the lowest point.

These steps are repeated iteratively, adjusting the model parameters opposite to the gradient direction until the algorithm converges to the minimum. The step size is controlled by a learning rate that defines how big the adjustments are at each iteration.


<br>

## 1 - [Gradient Descent (**Batch**) - Step by Step Code]()

Gradient Descent is an iterative algorithm to minimize a cost function by adjusting parameters opposite to the gradient direction. Batch Gradient Descent calculates the gradient using the entire dataset each step, resulting in stable but sometimes slow parameter updates.


<br>

#### [Cell 1]() - Setup and Data Generation

<br>


```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (intercept)

X_b = np.c_[np.ones((100, 1)), X]
```

<br>

#### [Cell 2]() - Gradient Descent Algorithm

<br>

```python
learning_rate = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)  \# random initialization

for iteration in range(n_iterations):
gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
theta = theta - learning_rate * gradients
```

<br>

#### [Cell 3]() - Output and Plotting

<br>

```python
print("Theta:", theta)

plt.plot(X, y, "b.")
X_new = np.array([, ])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta)
plt.plot(X_new, y_predict, "r-")
plt.title("Batch Gradient Descent")
plt.show()
```

<br><br>

## 2 [Stochastic Gradient Descent (SGD)]()

Stochastic Gradient Descent updates parameters based on a single random sample per iteration. This yields noisier but faster updates, suitable for large datasets and deep learning models.

<br>

#### [Cell 1]() - SGD Implementation

<br>

```python
theta = np.random.randn(2,1)  \# reset parameters
n_epochs = 50
t = 0  \# iteration counter for adaptive learning rate
m = len(X_b)

for epoch in range(n_epochs):
for i in range(m):
random_index = np.random.randint(m)
xi = X_b[random_index:random_index+1]
yi = y[random_index:random_index+1]
gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
eta = 0.1 / (1 + t * 0.01)  \# learning rate decay
theta = theta - eta * gradients
t += 1

print("Theta after SGD:", theta)
```

<br><br>

## 3. [Elastic Net Regularization]()

Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties to improve the performance of linear regression models, particularly when many variables are correlated. It helps prevent overfitting and performs automatic feature selection, making it a powerful tool for machine learning modeling.
<br>

#### [Cell 1]() - Data Splitting

<br>

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

<br>

#### [Cell 2]() - Elastic Net Model Training and Evaluation

<br>

```python

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=1000)
elastic_net.fit(X_train, y_train.ravel())

y_pred = elastic_net.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"ElasticNet Coefficients: {elastic_net.coef_}")
print(f"ElasticNet Intercept: {elastic_net.intercept_}")
print(f"Mean Squared Error: {mse}")
```

<br><br>


















































<br>
<br>

## [Other Algorithms Used to Train Machine Learning Models]()

- [**Batch Gradient Descent:**]() Computes the gradient using the entire dataset in each iteration, providing stable updates but slower on large datasets.

- [**Mini-batch Gradient Descent:**]() A compromise between batch and stochastic methods, using small batches of data to compute the gradient each step.

- [**Adam (Adaptive Moment Estimation):**]() An algorithm combining momentum and adaptive learning rates to improve convergence and training efficiency, especially in deep neural networks.

- [**RMSProp:**]() Adapts the learning rate for each parameter, useful to accelerate training and avoid oscillations.















































































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





