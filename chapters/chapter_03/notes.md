# Chapter 3 - A Tour of Machine Learning Classifiers Using Scikit-Learn

**Status:** In work  
**Code:** Perceptron and Adaline implementations  
**Focus:** Logistic Regression, Support Vector Machines, Decision Trees, k-Nearest Neighbors

## Summary

- *No free lunch theorem* states that no single classifier works best across all possible scenarios. It is recommended that you compare the performance of at least a handful of different learning algorithms to select the best model for the particular problem - these may differ in the number of features or examples, the amount of noise in a dataset, and whether the classes are linearly separable.  
- The 5 steps that are involved in training a supervised machine learning algorithm:  
  - Selecting features and collecting labeled training examples
  - Choosing a performance metric
  - Choosing a learning algorithm and training a model
  - Evaluating the performance of the model
  - Changing the settings of the algorithm and tuning the model
- Although many scikit-learn functions and class methods work with class labels in string format, using integer labels is a recommended approach to avoid technical glitches and improve computational performance due to smaller memory footprint; furthermore encoding class labels as integers is a common convention among most machine learning libraries.
- *Accuracy* is used often instead of missclassification error, and is calculated as $1-error$
- *Overfitting* means that the model captures patterns in the training data well but fails to generalize to unseen data.
- *Logistic regression* is a linear binary classification model that performs very well on linearly separable classes.
  - Logistic regression can be generalized to multiclass settings, also knowns as multinomal logistic regression, or softmax regression.  
  - *Odds* represents the odds in favor of a particular event.
  - *Logit* function is the logarithm of the odds (log-odds).  
  - We assume there is a linear relationship between the log-odds and the net inputs, but we are interested in the probability p. The logit function maps the probability to a real-number range, and the inverse function *logistic sigmoid function* maps the real-number range back to a [0, 1] range for probability p.
  - The output of the sigmoid function is the probability of a particular example belonging to class 1.
  - *Likelihood function* is the function we want to maximize when we build a logistic regression model. In practice, it is easier to maximize the log-likelihood since applying the logarithm reduces the potential for numerical underflows, and converting the product of factors into a summation makes it easier to calculate the derivative.
  - Logistic regression assumes the target variable comes from a Bernouilli distribution.
  - For minimizing the logistic regression loss function, it is recommended to use more advanced approaches than regular stochastic gradient descent. Scikit-learn has a solver parameter that implements some of these more advanced techniques, such as *limited-memory Broyden-Fletcher-Goldfarb-Shanno (BFGS)* algorithm, abbreviated as lbfgs.
- *Overfitting* is a common problem in machine learning, where a model performs well on the training data, but does not generalize well to unseen (test) data. If a model suffers from overfitting, we say it has a high *variance*, which can be caused by having too many parameters, leading to a model that is too complex given the underlying data. Similarly, a model can also suffer from *underfitting* (high bias), which means that our model is not complex enough to capture the pattern in the training data well and therefore also suffers from low performance on unseen data.  
- *Bias-variance tradeoff* refers to the perfromance of a model, that is, a model is either high variance (overfitting) or high bias (underfitting). Variance measures the consistency of the model prediction for classifying a particular example if we retrain the model multiple times on different subsets of the training dataset. We can say that the model is sensitive to the randomness in the training data. In contrast, bias measures how far off the predictions are from the correct values in general if we rebuild the model multiple times on different training datasets, bias is the measure of the systematic error that is not due to randomness.
- *Regularization* is a way to find a good bias-variance tradeoff via tuning the complexity of the model. Regularization is a very useful method for handling *collinearity* (high correlation among features), filtering out noise from data, and eventually preventing overfitting. 
- The concept behind regularization is to introduce additional information to penalize extreme parameter values. The most common form of regualarization is called L2 regularization.  
- Regularization is another reason why feature scaling (standardization) is important. For regularization to work properly, we need to ensure that all our features are on comparable scales.  
- *Support Vector Machine (SVM)* is another powerful and widely used algorithm that is an extension of the perceptron algorithm. In SVM, our optimization objective is to maximize the margin. The margin is defined as the distance between the separating hyperplane (decision boundary) and the training examples that are closest to this hyperplane, which are called *support vectors*.  
  - *Slack variables* serve to relax the linear constraints in SVM optimization objective for nonlinearly separable data to allow the convergence of the optimization in the presence of misclassifications, under appropriate loss penalization.
  - The use of slack variable introduces the variable known as C in SVM contexts. C is the hyperparameter for controlling the penalty for missclassification. Large values of C correspond to large error penalties, whereas we are less strict about misclassification errors if we choose smaller values for C. We can use C to control the width of the margin and therefore tune the bias-variance tradeoff.
  - In practical classification tasks, linear logistic regression and linear SVMs often yield very similar results. Logistic regression tries to maximize the conditional likelihoods of the training data, which makes it more prone to outliers than SVMs, which mostly care about the points that are closest to the decision boundary (support vectors). On the other hand, logistic regression has the advantage of being a simpler model and can be implemented more easily, and is mathematically easier to explain. Furthermore, logistic regression models can be easily updated, which is attractive when working with streaming data.
  - Another reason SVMs enjoy high popularity among machine learning practitioners is that they can be kernelized to solve nonlinear classification problems, commonly known as *kernel SVM*.
  - *Kernel methods* deal with linearly inseparable data by creating nonlinear combinations of the original features to project them onto a higher-dimensional space via a mapping function $\rho$ where the data becomes linearly separable.  
  - To solve a nonlinear problem using an SVM, we would transform the training data into a higher-dimensional feature space via a mapping function $\rho$ and train a linear SVM model to classify the data in this new feature space. Then, we can use the same function to transform new, unseen data to classify it using the linear SVM model.  
  - Construction of new features when we are dealing with high-dimensional data is computationally expensive, which makes us use *kernel trick*.
  - *Radial basis function (RBF)* kernel, also called *Gaussian kernel* is a similarity function between a pair of examples. The range of the function is from 0 to 1, and similar points get a score close to 1, while dissimilar points get a score close to 0.
- *Decision Tree* classifiers are attractive models if we care about interpretability. This model breaks down our data by making decisions based on asking a series of questions.
  - Based on the features in our training dataset, the decision tree model learns a series of questions to infer the class labels of the examples. Using the decision algorithm, we start at the tree root and split the data on the feature that results in the largest *information gain (IG)*. In an iterative process, we can then repeat this splitting procedure at each child node until the leaves are pure. This means that the training examples at each node all belong to the same class. In practice, this can result in a very deep tree with many nodes, which can easily lead to overfitting. Thus, we typically want to *prune* the tree by setting a limit for the maximum depth of the tree.
  - To split the nodes at the most informative features, we need to define an objective function to optimize via the tree learning algorithm. We use information gain as our objective function. The information gain is simply the difference between the *impurity* of the parent node and the sum of the child node impurities. The lower the impurities of the child nodes, the larger the information gain. For simplicity and to reduce the combinatorial search space, most libraries implement binary decision trees. 
  - The three impurity measures for splitting criteria that are commonly used in binary decision trees are *Gini impurity*, *entropy* and the *classification error*.  
  - The entropy is the negative of the sum of the proportions of the examples that belong to a class multiplied by the logarithm of this proportion, for each class. The entropy is 0 if all examples at a node belong to the same class, and the entropy is 1 (maximal) if we have uniform class distribution.  
  - The Gini impurity can be understood as a criterion to minimize the probability of misclassification. Similar to entropy, Gini impurity is maximal if the classes are perfectly mixed.  
  - In practice, both the Gini impurity and entropy typically yield very similar results, and it is not often worth spending much time on evaluating trees using different impurity criteria rather than experimenting with different pruning cut-offs.
  - Decision trees can build complex decision boundaries by dividing the feature space into rectangles. However, we have to be careful since the deper the decision tree, the more complex the decision boundary becomes, which can easily result in overfitting.  
  - It is important to note that feature scaling is not a requirement for decision tree algorithms.
- *Ensemble methods* have gained a huge popularity in applications of machine learning during the last decade due to their good classification performance and robustness toward overfitting. *Random forest* algorithm is a decision-tree based algorithm that is an ensemble of decision trees. The idea behind a random forest is to average multiple (deep) decision trees that individually suffer from high variance to build a more robust model that has a better generalization performance and is less susceptible to overfitting. The random forest algorithm can be summarized in four simple steps:
  - Draw a random *bootstrap* sample of size n (randomly choose n examples from the training dataset with replacement)
  - Grow a decision tree from the boostrap sample. At each node:  
    - Randomly select d features without replacement
    - Split the node using the feature that provides the best split according to the objective function, for instance, maximizing the information gain.
  - Repeat steps 1-2 k times
  - Aggregate the prediction by each tree to assign the class label by *majority vote*.
- Although random forests don't offer the same level of interpretability as decision treees, a big advantage of random forests is that we don't have to worry so much about choosing good hyperparameter values. We typically don't need to prune the random forest since the ensemble model is quite robust to the noise from averaging the predictions among the individual decision trees. The only parameter that we need to care about in practice is the number of trees, k, that we choose for the random forest. Typically, the larger the number of trees, the better the performance of the random forest classifier at the expense of an increased computational cost.
- Although it is less common in practice, the boostrap sample size and the number of features that are randomly chosen for each split are hyperparameters that can be tuned. Decreasing the size of the bootstrap sample increases the diversity among the individual trees since the probability that a particular training example is included in the bootstrap sample is lower. Thus, shrinking the size of the bootstrap samples may increase the randomness of the random forest, and it can help to reduce the effect of overfitting. However, smaller bootstrap samples typically result in a lower overall performance of the random forest and a small gap between trianing and test performance, but a low test performance overall. 
- In most implementations, including scikit-learn, the size of the bootstrap sample is chosen to be equal to the number of training examples in the original training dataset, which usually provides a good bias-variance tradeoff. For the number of features at each split, we want to choose a value that is smaller than the total number of features in the training dataset. A reasonable default that is used in scikit-learn and other implementations is $d = \sqrt(m)$, where m is the number of features in the dataset.  
- *k-nearest neighbor (KNN)*  is another supervised learning algorithm that is fundamentally different from the algorithms previously discussed. It is an example of a *lazy learner*, because it doesn't learn a discriminative function from the training data, and instead memorizes the training dataset.
  - *Parametric* and *non-parametric* are subgroups of machine learning models. Using parametric models, we estimate parameters from the training dataset to learn a function that can classify new data points without requiring the original training dataset anymore. Perceptron, logistic regression, and the linear SVM are parametric models. In contrast, non-parametric models can't be characterized by a fixed set of parameters, and the number of parameters changes with the amount of training data. Two examples of non-parametric training models that we have seen so far are the decision tree classifier/random forest and the kernel SVM. 
  - KNN belongs to the non-parametric model subgroup as it has an instance-based learning. Models based on instance-based learning memorize the training dataset, and lazy learning is a special case of instance-based learning that is associated with no cost during the learning process. 
  - The KNN algorithm can be summarized by the following steps:  
    - Choose the number of k and a distance metric
    - Find the k-nearest neighbors of the data record that we want to classify
    - Assign the class label by majority vote
  - The main advantage of such a memory-based approach is that the classifier immediately adapts as we collect new training data. However, the downside is that the computational complexity for classifying new examples grows linearly with the number of examples in the training dataset in the worst case scenario.  
  - In the case of a tie, the scikit-learn implementation of the KNN algorithm will prefer neighbors with a closer distance to the data record to be classified. If the neighbors have similar distances, the algorithm will choose the class label that comes first in the training dataset.
  - The right choice for k is crucial to finding a good balance between overfitting and underfitting. We also have to make sure that we chooes a distance metric that is appropriate for the features in the dataset. Often, a simple Euclidean distance measure is used for real-value examples. However, if we are using Euclidean distance, it is also important to standardize the data so that each feature contributes equally to the distance. The *minkowski distance* is a generalization of the p-norm distance, with p=2 corresponding to Euclidean distance.
  - It is important to mention that KNN is very susceptible to overfitting due to the *curse of dimensionality*. This is the phenomenon where the feature space becomes increasingly sparse for an increasing number of dimensions of a fixed-size training dataset. We can think of even the closest neighbors as being too far away in a high-dimensional space to give a good estimate. Feature selection and dimensionality reduction techniques are important to avoid curse of dimensionality.

## Key Terms/Formulas

Odds:

$$
\frac{p}{1-p}
$$

Logit:

$$
\mathrm{logit}(p) = \log\left(\frac{p}{1-p}\right) = \sum_{j=1}^{m} w_j x_j + b = \mathbf{w}^T \mathbf{x} + b
$$

Sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Threshold function:

$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \geq 0.5 \\
0 & \text{if } \hat{p} < 0.5
\end{cases}
$$

Likelihood function: 

$$
\mathcal{L}(w,b \mid x) = p(y \mid x; w,b) = \prod_{i=1}^{n} p\left(y^{(i)} \mid x^{(i)}; w,b\right) = \prod_{i=1}^{n}\left(\sigma\left(z^{(i)}\right)\right)^{y^{(i)}}\left(1-\sigma\left(z^{(i)}\right)\right)^{1-y^{(i)}}
$$

Log-likelihood function:

$$
\ell(w,b \mid x) = \log \mathcal{L}(w,b \mid x) = \sum_{i=1}^{n}\left[y^{(i)} \log\left(\sigma\left(z^{(i)}\right)\right) + \left(1-y^{(i)}\right)\log\left(1-\sigma\left(z^{(i)}\right)\right)\right]
$$

Updating the weights and bias for logistic regression:

$$
\begin{align}
\frac{\partial L}{\partial w_j} &= \frac{\partial L}{\partial \sigma}\frac{\partial \sigma}{\partial z}\frac{\partial z}{\partial w_j} \\
&= \frac{\sigma - y}{\sigma (1 - \sigma)} \times \sigma (1 - \sigma) \times x_j \\
&= -(y - \sigma)x_j \\
w_j &:= w_j + \eta(y - \sigma)x_j \\
b &:= b + \eta(y - \sigma)
\end{align}  
$$

L-2 regularization:

$$
\frac{\lambda}{2n}\lVert \mathbf{w} \rVert^2 = \frac{\lambda}{2n}\sum_{j=1}^{m}w_j^2
$$

Loss function for logistic regression with regularization: 

$$
\begin{align}
\ell(w,b) &= \frac{1}{n}\sum_{i=1}^{n}\left[-y^{(i)}\log\left(\sigma\left(z^{(i)}\right)\right) - \left(1-y^{(i)}\right)\log\left(1-\sigma\left(z^{i}\right)\right)\right] + \frac{\lambda}{2n}\lVert \mathbf{w} \rVert^2 \\
\frac{\partial \ell(w, b)}{\partial \w_j} = \left(\frac{1}{n} \sum_{i=1}^{n}\left(\sigma\left(\mathbf(w)^T\mathbf(x)^{(i)}\right)-y^{(i)}\right)x_j^{(i)}\right) + \frac{\lambda}{n}w_j
\end{align}
$$

Information gain:  

$$
IG\left(D_p, f\right) = I\left(D_p\right) - \frac{N_left}{N_p}I\left(D_left\right) - \frac{N_right}{N_p}I\left(D_right\right)
$$

Entropy:  

$$
I_h(t) = -\sum_{i=1}^{c}p(i\mid t)\log p(i\mid t)
$$

Gini impurity:

$$
I_G(t) = \sum_{i=1}^{c} p(i \mid t) \left(1 - p(i\mid t)\right) = 1 - \sum_{i=1}^{c} p(i\mid t)^2
$$

## Code work  
