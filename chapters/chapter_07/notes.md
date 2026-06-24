# Chapter 7 - Combining Different Models for Ensemble Learning

**Status:** In-Work  
**Code:** Bagging and Boosting  
**Focus:** Make predictions based on majority voting, use bagging to reduce overfitting by drawing random combinations of the training dataset with repetition, apply boosting to build powerful models from weak learners that learn from their mistakes

## Summary  

- The goal of *ensemble methods* is to combine different classifiers into a meta-classifier that has better generalization performance than each individual classifier alone. For example, assuming that we collected predictions from 10 experts, ensemble methods would allow us to strategically combine those predictions by the 10 experts to come up with a prediction that was more accurate and robust than the predictions by each individual expert.  
- A lot of ensemble methods use the *majority voting* principle. Majority voting simply means that we select the class label that has been predicted by the majority of the classifiers, that is, received more than 50 percent of the votes. Strictly speaking, the term majority vote refers to binary class settings only. However, it is easy to generalize the majority voting principle to multiclass settings, which is known as *plurality votinng*. Here, we select the class label that has the most votes.
- Using the training dataset, we start by training m different classifiers. Depending on the technique, the ensemble can be built from different classification algorithms, for example, decision trees, support vector machines, logistic regression classifiers, and so on. Alternatively, we can also use the same base classification algorithm, fitting different subsets of the training dataset. One prominent example of this approach is the random forest algorithm combining different decision tree classifiers.
- To predict a class label via simple majority or plurality voting, we can combine the predicted class labels of each individual classifier

## Key Terms/Formulas

Majority voting:

$\hat{y} = \text{mode}\{C_1(x), C_2(x), \dots, C_m(x)\}$  
