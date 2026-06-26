# Chapter 7 - Combining Different Models for Ensemble Learning

**Status:** In-Work  
**Code:** Bagging and Boosting  
**Focus:** Make predictions based on majority voting, use bagging to reduce overfitting by drawing random combinations of the training dataset with repetition, apply boosting to build powerful models from weak learners that learn from their mistakes

## Summary  

- The goal of *ensemble methods* is to combine different classifiers into a meta-classifier that has better generalization performance than each individual classifier alone. For example, assuming that we collected predictions from 10 experts, ensemble methods would allow us to strategically combine those predictions by the 10 experts to come up with a prediction that was more accurate and robust than the predictions by each individual expert.  
- A lot of ensemble methods use the *majority voting* principle. Majority voting simply means that we select the class label that has been predicted by the majority of the classifiers, that is, received more than 50 percent of the votes. Strictly speaking, the term majority vote refers to binary class settings only. However, it is easy to generalize the majority voting principle to multiclass settings, which is known as *plurality votinng*. Here, we select the class label that has the most votes.
- Using the training dataset, we start by training m different classifiers. Depending on the technique, the ensemble can be built from different classification algorithms, for example, decision trees, support vector machines, logistic regression classifiers, and so on. Alternatively, we can also use the same base classification algorithm, fitting different subsets of the training dataset. One prominent example of this approach is the random forest algorithm combining different decision tree classifiers.
- To predict a class label via simple majority or plurality voting, we can combine the predicted class labels of each individual classifier and select the class label $\hat{y}$ that received the most votes.
- To illustrate why ensemble methods can work better than individual classifiers alone, let's apply some concepts of combinatorics. For the following example, we will make the assumption that all n-base classifiers for a binary classification task have an equal error rate $\epsilon$. Furthermore, we will assume that the classifiers are independent and the error rates are not correlated. Under those assumptions, we can simply express the error probability of an ensemble of base classifiers as a probability mass function of a binomial distribution. If we take a look at a concrete example of 11 base classifiers where each classifier has an error rate of 0.25, the error rate of the ensemble is calculated to be 0.034, much lower than the error rate of each individual classifier, if all the assumptions are made.
- Weighted majority voting builds on majority voting and includes a weight for each classifier. There is also a modified version that predicts class labels from probabilities, which is useful when the classifiers in our ensemble are well calibrated.
- The majority vote approach should not be confused with stacking. The stacking algorithm can be understood as a two-level ensemble, where the first level consists of individual classifiers that feed their predictions to the second level, where another classifier is fit to the level-one classifier predictions to make the final predictions.
- *Bagging* is an ensemble learning technique that is closely realted to the majority voting. However, instead of using the same training dataset to fit the individual classifiers in the ensemble, we draw bootstrap samples (random samples with replacement) from the initial training dataset, which is why bagging is also known as *bootstrap aggregating*. Each classifier receives a random subset of examples from the training dataset. We denote these random samples obtained via bagging as Bagging round 1, Bagging round 2, and so on. Each subset contains a certain portion of duplicates and some of the original examples don't appear in a resampled dataset at all due to sampling with replacement. Once the individual classifiers are fit to the bootstrap samples, the predictions are combined using majority voting.  
- Random forests are a special case of bagging where we also use random feature subsets when fitting the individual decision trees.
- In practice, complex classification tasks and a dataset's high dimensionality can easily lead to overfitting in single decision trees, and this is where the bagging algorithm can really play to its strengths. We also have to note the bagging algorithm can be an effective approach to reducing the variance of a model. However, bagging is ineffective in reducing model bias, that is, models that are too simple to capture the trends in the data well. This is why we want to perform bagging on an ensemble of classifiers with low bias, for example, unpruned decision trees.
- In *boosting*, the ensemble consists of very simple base classifiers, also often referred to as *weak learners*, which often only have a slight performance advantage over random guessing (e.g. decision tree stump). The key concept behind boosting is to focus on training examples that are hard to classify, that is, to let the weak learners subsequently learn from misclassified training examples to improve the performance of the ensemble.  
- In contrast to bagging, the initial formulation of the boosting algorithm uses random subsets of training examples drawn from the training dataset without replacement, the original boosting procedure can be summarized in the following four steps:
  - Draw a random subset of training examples, without replacement from the training dataset, to train a weak learner
  - Draw a second random training subset wihtout replacement from the training dataset and add 50 percent of the examples that were previously misclassified to train a weak learner
  - Find the training examples in the training dataset which the previous two weak learners disagree upon, to train a third weak learner
  - Combine the weak learners via majority voting
- Boosting can lead to a decrease in bias as well as variance compared to bagging models. In practice, however, boosting algorithms such as *AdaBoost* are also known for their high variance, that is, the tendency to overfit the training data. In contrast to the original boosting procedure, AdaBoost uses the complete training dataset to train the weak learners, where the training examples are reweighted in each iteration to build a strong classifier that learns from the mistakes of the previous weak learners in the ensemble. The AdaBoost algorithm works as follows:
  - Set the weight vector $\mathbf{w}$ to uniform weights, where $\sum_{i} w_i  = 1$
  - For j in m boosting rounds, do the following:
    - Train a weighted weak learner: $C_j = train(X, y, w)$
    - Predict class labels $\mathbf{\hat{y}} = predict(C_j, X)$
    - Compute the weighted error rate: $\epsilon = \mathbf{w} \dot (\hat{\mathbf{y}} \neq \mathbf{y})$
    - Compute the coefficient: $\alpha_{j} = 0.5 log \frac{1-\epsilon}{\epsilon}$
    - Update the weights: $\mathbf{w} := \mathbf{w} \times exp(-\alpha_{j} \times \hat{\mathbf{y}} \times \mathbf{y})$
    - Normalize the weights to sum to 1: $\mathbf{w} := \mathbf{w}/\sum_{i} w_i$
  - Compute the final prediction: $\hat{\mathbf{y}} = (\sum_{j=1}^m(\alpha_j \times predict(C_j, \mathbf{X})) > 0$

## Key Terms/Formulas

Majority voting:

$$`
\hat{y} = \mathrm{mode}\lbrace C_1(x), C_2(x), \dots, C_m(x)\rbrace`
$$

Error probability of an ensemble of base classifiers:

$$
P(y \geq k) = \sum_{k}^{n}\binom{n}{k} \epsilon^k (1 - \epsilon)^{n-k} = \epsilon_{\text{ensemble}}
$$  

Weighted majority voting:

$$
\hat{y} = \underset{i}{\arg\max} \sum_{j=1}^{m} w_j \chi_A(C_j(\mathbf{x}) = i)
$$

Weighted majority voting using class probabilities:

$$
\hat{y} = \underset{i}{\arg\max} \sum_{j=1}^{m} w_j p_{ij}
$$
