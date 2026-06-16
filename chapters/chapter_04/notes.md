# Chapter 4 - Building Good Training Datasets - Data Preprocessing  

**Status:** Completed  
**Code:** Data Preprocessing  
**Focus:** Removing and imputing missing values from a dataset, getting categorical data into shape, selecting relevant features for model construction  

## Summary  

- The quality of the data and the amount of information it contains are key factors that determine how well a machine learning algorithm can learn.  
- It is not uncommon for real-world applications to be missing one or more values for various reasons. Errors in data collection process, certain measures being unapplicable, fields being left blank in a survey etc.  
- Most computational tools are unable to handle missing values or will produce unpredictable results if we ignore them.  
- When using the scikit-learn API, it is recommended to use the underlying numpy arrays of the pandas dataframes by using the values attribute before feeding it to the estimators.  
- Although removing missing data is convenient, we may end up removing too many samples, making a reliable analysis impossible. If we remove too many feature columns, we risk losing valuable information that our classifier needs to discriminate between classes.  
- We can use different interpolation techniques to estimate the missing values from the other training examples in our dataset. One of the most common interpolation techniques is *mean imputation*, where we simply replace the missing value with the mean value for the entire feature column.  
- Other options for the interpolation are median or most frequent. This is useful for computing categorical features, for example, a feature column that stores an encoding of color names, such as red, green, and blue.  
- When we are working with categorical data, it is important to distinguish between *ordinal* and *nominal* features. Ordinal features can be understood as categorical values that can be sorted or ordered. In contrast, nominal features don't imply any order.
- Many machine learning libraries require that the class labels are encoded as integer values. Although most estimators for classification in scikit-learn convert class labels to integers internally, it is considered good practice to provide class labels as integer arrays to avoid technical glitches.  
- One of the most common mistakes in dealing with categorical data is creating an order when converting nominal data to integers (e.g. assigning blue = 0, green = 1, red = 2 makes the model think green is larger than blue)
- *One-hot encoding* gets around this problem by creating a new dummy feature for each unique value in the nominal feature column. It uses binary values to indicate the presence of a particular value.
- When we are using one-hot encoding datasets, we have to keep in mind that this introduces multi-collinearity, which can be an issue for certain methods (for instance, methods that require matrix inversion). If features are highly correlated, matrices are computationally difficult to invert, which can lead to numerically unstable estimates. To reduce the correlation among the variables, we can simply remove one feature column from the one-hot encoded array.  
- While one-hot encoding is the most common way to encode unordered categorical variables, several alternative methods exist. Some of these techniques can be useful when working with categorical features that have high cardinality (a large number of unique category labels).
  - *Binary encoding* produces multiple binary features but requires fewer feature columns (i.e. log(K) instead of K-1) where K is the number of unique categories. The numbers are first converted to binary representations, and each binary number position will form a new feature column.
  - *Count (frequency) encoding* replaces the label of each category by the number of times or frequency it occurs in the training set.
- If we are dividing the dataset into training and test datasets, we have to keep in mind that we are withholding valuable information that the learning algorithm could benefit from. However, the smaller the test dataset, the more inaccurate the estimation of the generalization error. In practice, the most commonly used splits are 60:40, 70:30, or 80:20, depending on the size of the initial dataset. However, for large datasets, 90:10 or 99:1 splits are also common and appropriate.  
- The majority of machine learning and optimization algorithms behave much better if the features are on the same scale.  
- There are two common approaches to bringing different features onto the same scale:
  - *Normalization* refers to the rescaling of the features to a range of [0, 1], which is a special case of *min-max scaling*. To normalize our data, we can simply apply the min-max scaling to each feature column.
  - *Standardization* is more practical more many machine learning algorithms, especially for optimization algorithms like gradient descent. The reason is that many models initialize the weights to 0 or small random values close to 0, and when we center feature columns at a mean of 0 with a standard deviation of 1 (standard normal distribution), it makes it easier to learn the weights. Standardization does not change the shape of the distribution and does not transform a non-normally distributed data into normally distributed data. It also maintains useful information about outliers and makes algorithms less sensitive.
- If we notice that a model performs much better on the training dataset than on the test dataset, this observation is a strong indicator of overfitting. We say the model has *high variance* The reason for overfitting is that our model is too complex for the given training data. Common solutions to reduce the generalization error are as follows:
  - Collect more training data
  - Introduce a penalty for complexity via regularization
  - Choose a simpler model with fewer parameters
  - Reduce the dimensionality of the data
- *Dimensionality reduction* via feature selection is especially useful for unregularized models for reducing complexity and avoiding overfitting. There are two main categories of dimensionality reduction: *feature selection* and *feature extraction*. Via feature selection, we select a subset of the original features, whereas in feature extraction, we derive information from the feature set to construct a new feature subspace.
- Sequential feature selection algorithms are a family of greedy search algorithms that are used to reduce an initial d-dimensional feature space to a k-dimensional feature subspace where k < d. The motivation behind feature selection algorithms is to automatically select a subset of features that are most relevant to the problem, to improve computational efficiency, or to reduce the generalization error of the model by removing irrelevant features or noise.
- A classic sequential feature selection algorithm is *sequential backward selection (SBS)* which aims to reduce the dimensionality of the initial feature subspace with a minimum decay in performance of the classifier to improve upon computational efficiency. In certain cases, SVS can even improve the predictive power of the model if a model suffers from overfitting.
- The idea behind SBS is quite simple, SBS sequentially removes features from the full feature subset until the new feature subspace contains the desired number of features. To determine which feature is to be removed at each stage, we need to define the criterion function that we want to minimize.  
- The criterion calculated by the criterion function can simply be the difference in performance of the classifier before and after the removal of a particular feature. Then, the feature to be removed at each stage can simply be defined as the feature that maximizes this criterion, or in more simple terms, at each stage we eliminate the feature that causes the least performance loss after removal.
- We can also measure the feature importance as the averaged impurity decrease computed from all decision trees in the random forest, without making assumptions about whether our data is linearly separable or not. However, as far as interpretability is concerned, the random forest technique comes with an important gotcha that is worth mentioning. If two or more features are highly correlated, one feature may be ranked very highly while the information on the other features(s) may not be fully captured. On the other hand, we don't need to be concerned about this problem if we are merely interested in the predictive performance of the model rather than the interpretation of feature importance values.

## Key Terms/Formulas

Min-max scaling:

$x_{norm}^{(i)} = \frac{x^{i} - x_{min}}{x_{max} - x_{min}}$

Standardization:

$x_{std}^{(i)} = \frac{x^{i} - \mu{x}}{\sigma_{x}}$
