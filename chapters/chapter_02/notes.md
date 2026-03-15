# Chapter 2 - Training Simple Machine Learning Algorithms for Classification

**Status:** Complete  
**Code:** Perceptron and Adaline implementations  
**Focus:** Perceptron, Adaline, Gradient Descent

## Summary

- The perceptron algorithm:  
  - Initialize the weights and bias to 0 or small random numbers
  - For each training example $x^{(i)}$:  
    - Compute the output value $\hat{y}^{(i)}$
    - Update the weights and bias unit  
  - The convergence of perceptron algorithm is only guaranteed if the two classes are linearly separable.
  - While it is possible to initialize the weights to 0, if we did that the learning rate would have no effect on the decision boundary. It would only affect the scale of the weight vector and not the direction.  
- *One-versus-all (OvA)* or *one-versus-rest (OvR)* is a technique that allows us to extend any binary classifier to multi-class problems. Using OvA, we can train one classifier per class, where the particular class is treated as the positive class and the examples from all other classes are considered negative classes. If we were to classify a new, unlabeled data instance, we would use our n classifiers, where n is the number of class labels, and assign the class label with the highest confidence to the particular instance we want to classify.  
- The Adaline algorithm:  
  - The key difference between the Adaline and perceptron is that the weights are updated based on a linear activation function rather than a unit step function like in the perceptron.
- Gradient Descent:
  - One of the key ingredients of supervised machine learning algorithms is the objective function (also known as loss function or cost function) to be optimized (minimized) during the learning process.  
  - In the case of Adaline, the loss function L is the mean squared error (MSE) between the calculated outcome and the true class label.
  - This loss function is differentiable and convex, so we can use *gradient descent* to find the weights that minimize it.
  - In gradient descent, in each iteration, we take a step in the opposite direction of the gradient, where the step size is determined by the value of the learning rate, as well as the slope of the gradient.
  - *Full batch gradient descent* is referred to the approach of updating weights based on all examples in the training dataset (instead of updating the parameters incrementally after each training example)
  - *Standardization* helps gradient descent converge faster. It becomes easier to find a learning rate that works well for all weights if the features are on similar scales. Otherwise, a learning rate that works well for updating one weight might be too large or too small to update the other weight equally well.  
  - *Stochastic Gradient Descent* (also called iterative or online gradient descent) updates the parameters incrementally for each training example instead of updating them based on the sum of the accumulated errors over all training examples. Although it can be considered as an approximation of gradient descent, it typically reaches convergence much faster bceause of the mroe frequent weight updates. Since each gradient is calculated based on a single training example, the error surface is noisier than in gradient descent, which can also have the advantage that SGD can escape shallow local minima more readily in nonlinear loss functions. To obtain good results, present the dataset in random order and shuffle for every epoch to avoid cycles.  
  - *Online learning* involves models training on the fly as new training data arrives. It is useful if we are accumulating large amounts of data, (e.g. customer data in web applications). SGD works well in online learning, the system can immediately adapt to changes, and the training data can be discarded after updating the model if storage space is an issue.
  - Adaptive learning rate that decreases over time is preferred in SGD over fixed learning rate.
  - *Mini-batch gradient descent* is a compromise between full batch gradient descent and SGD. It's applying full batch gradient descent to smaller subsets of the training data (e.g. 32 training examples at a time). It converges faster than full batch gradient descent due to more frequent weight updates.  
- The learning rate and the number of epochs are *hyperparameters* of the perceptron and Adaline algorithms.

## Key Terms/Formulas

$$
\begin{aligned}
    w_j &:= w_j + \Delta w_j \\
    b &:= b + \Delta b \\
    \Delta w_j &= \eta(y^{(i)} - \hat{y}^{(i)})x_j^{(i)} \\
    \Delta b &= \eta(y^{(i)} - \hat{y}^{(i)})
\end{aligned}
$$

$$
\Delta \mathbf{w} = -\eta \nabla_{\mathbf{w}} L(w,b), \qquad
\Delta b = -\eta \nabla_b L(w,b)
$$

$$
\frac{\partial L}{\partial w_j} = -\frac{2}{n}\sum_{i=1}^{n}\left(y^{(i)} - \sigma\left(z^{(i)}\right)\right)x_j^{(i)}
$$

$$
\frac{\partial L}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}\left(y^{(i)} - \sigma\left(z^{(i)}\right)\right)
$$

$$
\Delta w_j = -\eta \frac{\partial L}{\partial w_j}, \qquad
\Delta b = -\eta \frac{\partial L}{\partial b}
$$

$$
\mathbf{w} := \mathbf{w} + \Delta \mathbf{w}, \qquad b := b + \Delta b
$$

## Code work

- Reproduced: Implementations for the perceptron and Adaline algorithms using gradient and stochastic gradient descent
- Modified: The decision boundary plotting function was written to be more clear
