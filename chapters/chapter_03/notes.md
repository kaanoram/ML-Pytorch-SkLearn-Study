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
- *Accuracy* is used often instead of missclassification error, and is calculated as $ 1 - error $
- *Overfitting* means that the model captures patterns in the training data well but fails to generalize to unseen data.
- *Logistic regression* is a linear binary classification model that performs very well on linearly separable classes.
  - Logistic regression can be generalized to multiclass settings, also knowns as multinomal logistic regression, or softmax regression.  
  - *Odds* represents the odds in favor of a particular event.
  - *Logit* function is the logarithm of the odds (log-odds).  
  - We assume there is a linear relationship between the log-odds and the net inputs, but we are interested in the probability p. The logit function maps the probability to a real-number range, and the inverse function *logistic sigmoid function* maps the real-number range back to a [0, 1] range for probability p.
  - The output of the sigmoid function is the probability of a particular example belonging to class 1.

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

## Code work  
