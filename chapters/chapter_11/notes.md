# Chapter 11 - Implementing a Multilayer Artificial Neural Network from Scratch

**Status:** Completed  
**Code:** Neural Networks  
**Focus:** Gaining a conceptual understanding of multilayer NNs, implementing the fundamental backpropagation algorithm for NN training from scratch, training a basic multilayer NN for image classification  

## Summary

- The units in the **hidden layer** are fully connected to the input features, and the **output layer** is fully connected to the hidden layer. If such a network has more than one hidden layer, we also call it a **deep NN**.
- **Multilayer perceptron (MLP)** is a fully connected, multilayer feedforward NN. We can add any number of hidden layers to the MLP to create deeper network architectures. Practically, we can think of the number of layers and units in an NN as additional hyperparameters that we want to optimize for a given problem task using cross-validation technique. However, the loss gradients for updating the network's parameters, which we calculate via backpropagation, will become increasingly small as more layers are added to a network. This **vanishing gradient** problem makes model learning more challenging. Therefore, special algorithms have been developed to help train such DNN structures, this is known as **deep learning**.
- While one unit in the output layer would suffice for a binary classification task, a more general form of an NN allows us to perform multiclass classification via a generalization of the **one-versus-all (OvA)** technique. We can use the **one-hot** representation of the categorical variables to tackle classification tasks with an arbitrary number of unique class labels present in the training dataset.
- The process of **forward propagation** to calculate the output of an MLP model can be summarized in three steps:
  - Starting at the input layer, we forward propagate the patterns of the training data through the network to generate an output
  - Based on the network's output, we calculate the loss that we want to minimize using a loss function.
  - We backpropogate the loss, find its derivative with respect to each weight and bias unit in the network, and update the model.
- Finally, after we repeat these three steps for multiple epochs and learn the weight and bias parameters of the MLP, we use forward propagation to calculate the network output and apply a threshold function to obtain the predicted class labels in the one-hot representation, which we described in the previous section.
- The activation function $\sigma$ needs to be differentiable to learn the weights that connect the neurons using a gradient-based approach. To be able to solve complex problems such as image classification, we need nonlinear activation functions in our MLP model, for example, the sigmoid (logistic) activation function. The sigmoid function is an S-shaped curve that maps the net input z onto a logistic distribution in the range 0 to 1.
- MLP is a typical example of a feedforward artificial NN. The term **feedforward** refers to the fact that each layer serves as the input to the next layer without loops, in contrast to recurrent NNs. The term multilayer perceptron may sound a little bit confusing since the artificial neurons in this network architecture are typically sigmoid units, not perceptrons. We can think of the neurons in the MLP as logistic regression units that return values in the continous range between 0 and 1.
- One way to decrease the effect of overfitting is to increase the regularization strength via L2 regularization. Another useful technique for tackling overfitting in NNs is dropout.
- Some tricks to fine-tune NNs are:
  - Adding skip-connecitons, which are the main contribution of residual NNs
  - Using learning rate schedulers that change the learning rate during training
  - Attaching loss functions to earlier layers int he networks as it's being done in the popular Inception v3 architecture
- For MSE loss, we have to sum or average over the t activation units in our network in addition to averaging over the n examples in the dataset or mini-batch. Our goal is to minimize the loss function $L(\mathbf{W})$, thus we need to calculate the partial derivative of the parameters $\mathbf{W}$ with respect to each weight for every layer in the network.
- Note that $\mathbf{W}$ consists of multiple matrices. In an MLP with one hidden layer, we have the weight matrix, $\mathbf{W}^{(h)}$ which connects the input to the hidden layer, and $\mathbf{W}^{(out)}$ which connects the hidden layer to the output layer. It may seem that both $\mathbf{W}^{(h)}$ and $\mathbf{W}^{(out)}$ have the same number of rows and columns, but this is typically not the case unless we initialize an MLP with the same number of hidden units, output units, and input features.
- In essence we can think of **backpropagation** as a very computationally efficient approach to compute the partial derivatives of a complex, non-convex loss function in multilayer NNs. Here, our goal is to use those derivatives to learn the weight coefficients for parameterizing such a multilayer NNs. The challenge in the parameterization of NNs is that we are typically dealing with a very large number of model parameters in a high-dimensional feature space. In contrast to loss functions of single-layer NNs such as Adaline or logistic regression, the error surface of an NN loss function is not convex or smooth with respect to the parameters. There are many bumps in this high-dimensional loss surface (local minima) that we have to overcome in order to find the global minimum of the loss function.
- In the context of computer algebra, a set of techniques, known as **automatic differentiation**, has been developed to solve the chain rule for an arbitrary long function composition. Automatic differentiation comes with two modes, the forward and the reverse modes; backpropagation is simply a special case of reverse-mode automatic differentiation. The key point is that applying the chain rule in forward mode could be quite expensive since we would have to multiply large matrices for each layer (Jacobians) that we would eventually multiply by a vector to obtain the output. The trick of the reverse mode is that we traverse the chain rule form right to left. We multiply a matrix by a vector, which yields another vector that is multiplied by the next matrix, and so on. Matrix-vector multiplication is computationally much cheaper than matrix-matrix multiplication, which is why backpropagation is one of the most popular algorithms used in NN training.
- In backpropagation, we propagate the error from right to left. We can think of this as an application of the chain rule to the computation of the forward pass to compute the gradient of the loss with respect to the model weights (and bias units).
- To compute the partial derivative, which is used to update $w_{1,1}^{(out)}$, we can compute the three individual partial derivative terms and multiply the results. For simplicity, we will omit averaging over the individual examples in the mini-batch.
- You might be wondering why we did not use regular gradient descent but instead used mini-batch learning to train our NN for the handwritten digit classification exercise. In online learning, we compute the gradient based on a single training example at a time to perform the weight update. Although this is a stochastic approach, it often leads to very accurate solutions with a much faster convergence than regular gradient descent. Mini-batch learning is a special form of SGD where we compute the gradient based on a subset of k of the n training samples. Mini-batch learning has an advantage over online learning in that we can make use of our vectorized implementations to improve the computational efficiency. However, we can update the weights much faster than in regular gradient descent. Intuitively, you can think of mini-batch learning as predicting the voter turnout of a presidential election from a poll by asking only a representative subset of the population rather than asking the entire population.
  
### Key Terms/Formulas

Activation unit of the hidden layer:

$$
z_1^{(h)} = x_1^{(in)}w_{1,1}^{(h)} + x_2^{(in)}w_{1,2}^{(h)} + \cdots + x_m^{(in)}w_{1,m}^{(h)} \\
a_1^{(h)} = \sigma(z_1^{(h)})
$$

Sigmoid activation function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

MSE loss for NN:

$$
L(\mathbf{W}, \mathbf{b}) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{t} \sum_{j=1}^{t} (y_j^{[i]} - a_j^{(out)[i]})^2
$$

Forward propagation:

$$
\mathbf{Z}^{(h)} = \mathbf{X}^{(in)}\mathbf{W}^{(h)T} + \mathbf{b}^{(h)} \text{(net input of the hidden layer)} \\
\mathbf{A}^{(h)} = \sigma(\mathbf{Z}^{(h)}) \text{(activation of the hidden layer)}\\
\mathbf{Z}^{(out)} = \mathbf{A}^{(h)}\mathbf{W}^{(out)T} + \mathbf{b}^{(out)} \text{(net input of the output layer)}\\
\mathbf{A}^{(out)} = \sigma(\mathbf{Z}^{(out)}) \text{(activation of the output layer)}
$$

Gradient for output layer weight:

$$
\frac{\partial L}{\partial w_{1, 1}^{(out)}} = \frac{\partial L}{\partial a_1^{(out)}}\frac{\partial a_{1}^{(out)}}{\partial w_{1, 1}^{(out)}}
$$

The partial derivative of the MSE loss with respect to the predicted output score of the first output node:

$$
\frac{\partial L}{\partial a_1^{(out)}} = \frac{\partial}{\partial a_1^{(out)}}(y_1 - a_1^{(out)})^2 = 2(a_1^{(out)} - y)
$$

The derivative of the logistic sigmoid function:

$$
\frac{\partial a_1^{(out)}}{\partial z_1^{(out)}} = \frac{\partial}{\partial z_1^{(out)}}\frac{1}{1 + e^{z_1^{(out)}}} = a_1^{(out)}(1 - a_1^{(out)})
$$

Derivative of the net input with respect to the weight:

$$
\frac{\partial z_1^{(out)}}{\partial w_{1, 1}^{(out)}} = \frac{\partial}{\partial w_{1, 1}^{(out)}}a_1^{(h)}w_{1, 1}^{(out)} + b_{1}^{(out)} = a_1^{(h)}
$$

The gradient of the loss function with respect to the output layer weight:

$$
\frac{\partial L}{\partial w_{1, 1}^{(out)}} = \frac{\partial L}{\partial a_1^{(out)}}\frac{\partial a_1^{(out)}}{\partial z_1^{(out)}}\frac{\partial z_{1}^{(out)}}{\partial w_{1, 1}^{(out)}}
$$

The gradient of the loss function with respect to the hidden layer weight:

$$
\frac{\partial L}{\partial w_{1, 1}^{(h)}} = \frac{\partial L}{\partial a_1^{(out)}}\frac{\partial a_1^{(out)}}{\partial z_1^{(out)}}\frac{\partial z_1^{(out)}}{\partial a_1^{(h)}}\frac{\partial a_1^{(h)}}{\partial z_1^{(h)}}\frac{\partial z_1^{(h)}}{\partial w_{1, 1}^{(h)}} + \frac{\partial L}{\partial a_2^{(out)}}\frac{\partial a_2^{(out)}}{\partial z_2^{(out)}}\frac{\partial z_2^{(out)}}{\partial a_1^{(h)}}\frac{\partial a_1^{(h)}}{\partial z_1^{(h)}}\frac{\partial z_1^{(h)}}{\partial w_{1, 1}^{(h)}}
$$
