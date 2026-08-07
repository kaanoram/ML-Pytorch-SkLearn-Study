# Chapter 12 - Parallelizing Neural Network Training with PyTorch

**Status:** Completed  
**Code:** Neural Networks  
**Focus:** How PyTorch improves training performance, working with PyTorch's Dataset and DataLoader to build input pipelines and enable efficient model training, working with PyTorch to write optimized machine learning code, using the torch.nn module to implement common deep learning architectures conveniently, choosing activation functions for artificial NNs  

## Summary

- Many functions in scikit-learn allow us to spread those computations over multiple processing units. However, by default, Python is limited to execution on one core due to the **global interpreter lock (GIL)**. So, although we indeed take advantage of Python's multiprocessing library to distribute our computations over multiple cores, we still have to consider that the most advanced desktop hardware rarely comes with more than 8 or 16 such cores.
- In order to work with neural networks with large amounts of parameters, it is a good idea to use **graphics processing units (GPUs)**. You can think of a graphics card as a small computer inside your machine. At 2.2 times the price of a modern CPU, we can get a GPU that has 640 times more cores and is capable of around 46 times more floating-point calculations per second. So, what is holding us back from utilizing GPUs for our machine learning tasks? The challenge is that writing code to target GPUs is not as simple as executing Python code in our interpreter. There are special packages, such as CUDA and OpenCL, that allow us to target the GPU. However, writing code in CUDA or OpenCL is probably not the most convenient way to implement and run machine learning algorithms. This is what PyTorch was developed for.
- PyTorch is a scalable and multiplatform programming interface for implementing and running machine learning algorithms, including convenience wrappers for deep learning.
- To improve the performance of training machine learning models, PyTorch allows execution on CPUs, GPUs, and XLA devices such as TPUs. However, its greatest performance capabilities can be discovered when using GPUs and XLA devices. PyTorch supports CUDA-enabled ROCm GPUs officially.
- PyTorch is built around a computation graph composed of a set of nodes. Each node represents an operation that may have zero or more inputs or outputs. PyTorch provides an imperative programming environment that evaluates operations, executes computation, and returns concrete values immediately. Hence, the computation graph in PyTorch is defined implicitly, rather than constructed in advance and executed after.
- Mathematically, tensors can be understood as a generalization of scalars, vectors, matrices, and so on. More concretely, a scalar can be defined as a rank-0 tensor, a vector can be defined as a rank-1 tensor, a matrix can be defined as a rank-2 tensor, and matrices stacked in a third dimension can be defined as rank-3 tensors. Tensors in PyTorch are similar to NumPy's arrays, except that tensors are optimized for automatic differentiation and can run on GPUs.
- When we are training a deep NN model, we usually train the model incrementally using an iterative optimization algorithm such as stochastic gradient descent. In cases where the training dataset is rather small and can be loaded as a tensor into the memory, we can directly use this tensor for training. In typical use cases, however, when the dataset is too large to fit into the computer memory, we will need to load the data from the main storage device (for example, the hard drive or solid-state drive) in chunks, that is, batch by batch. In addition, we may need to construct a data-processing pipeline to apply certain transformations and preprocessing steps to our data, such as mean centering, scaling, or adding noise to augment the training procedure and to prevent overfitting.
- Applying preprocessing functions manually can be quite cumbersome. PyTorch provides a special class for constructing efficient and convenient preprocessing pipelines.
- For simplicity, we have only discussed the sigmoid activation function in the context of multilayer feedforward NNs so far. Technically, we can use any function as an activation function in multilayer NNs as long as it is differentiable. We can even use linear activation functions, such as in Adaline. However, in practice, it would not be very useful to use linear activation functions for both hidden and output layers, since we want to introduce nonlinearity in a typical artificial NN to be able to tackle complex problems. The sum of linear functions yields a linear function after all.
- The logistic (sigmoid) activation function probably mimics the concept of a neuron in brain most closely - we can think of it as the probability of whether a neuron fires. However, the logistic activation function can be problematic if we have highly negative input, since the output of the sigmoid function will be close to zero in this case. If the sigmoid function returns output that is close to zero, the NN will learn very slowly, and it will be more likely to get trapped in the local minima of the loss landscape during training. This is why people often prefer a hyperbolic tangent as an activation function in hidden layers.
- The softmax function is a soft form of the argmax function, instead of giving a single class index, it provides the probability of each class. Therefore, it allows us to compute the meaningful class probabilities in multiclass settings (multinomial logistic regression). In softmax, the probability of a particular sample with next input $z$ belonging to the $i$th class can be computed with a normalization term in the denominator, that is, the sum of the exponentially weighted linear functions.
- Another sigmoidal function that is often used in the hidden layers of artificial NNs is the **hyperbolic tangent (tanh)**, which can be interpreted as a rescaled version of the logistic function. The advantage of the hyperbolic tangent over the logistic function is that it has a broader output spectrum ranging in the open interval $(-1, 1)$, which can improve the convergence of the backpropagation algorithm. In contrast, the logistic function returns an output signal ranging in the open interval $(0, 1)$.
- The **rectified linear unit (ReLU)** is another activation function that is often used in deep NNs. Before we delve into ReLU, we should step back and understand the vanishing gradient problem of tanh and logistic activations. To understand this problem, let's assume that we initially have the net input $z_1 = 20$, which changes to $z_2 = 25$. Computing the tanh activation, we get $\sigma(z_1) = 1.0$ and $\sigma(z_2) = 1.0$, which shows no change in the output (due to the asymptotic behavior of the tanh function and numerical errors). This means that the derivative of activations with respect to the net input diminishes as $z$ becomes large. As a result, learning the weights of during the training phase becomes very slow because the gradient terms may be very close to zero. ReLU activation addresses this issue. ReLU is still a nonlinear function that is good for learning complex functions with NNs. Besides this, the derivative of ReLU, with respect to its input, is always 1 for positive input values. Therefore, it solves the problem of vanishing gradients, making it suitable for deep NNs.

### Key Terms/Formulas

Softmax:

$$
p(z) = \sigma(z) = \frac{e^{z_i}}{\sum_{j=1}^{M} e^{z_j}}
$$

Hyperbolic tangent:

$$
\sigma_{tanh}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

Rectified Linear Unit (ReLU):

$$
\sigma(z) = max(0, z)
$$
