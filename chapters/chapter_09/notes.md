# Chapter 9 - Predicting Continuous Target Variables with Regression Analysis

**Status:** Completed  
**Code:** Regression Analysis  
**Focus:** Exploring and visualizing datasets, looking at different approaches to implementing linear regression models, training regression models that are robust to outliers, evaluating regression models and diagnosing common problems, fitting regression models to nonlinear data

## Summary

- The goal of linear regression is to model the relationship between one or multiple features and a continuous target variable. In contrast to classification, a different subcategory of supervised learning, regression analysis aims to predict outputs on a continuous scale rather than categorical class labels.
- The goal of **simple (univariate) linear regression** is to model the relationship between a single feature (**explanatory variable**, x) and a continuous-valued **target (response variable, y)**. The parameter b (bias) represents the y axis intercept and $w_1$ is the weight coefficient fo the explanatory variable. Our goal is to learn the weights of the linear equation to describe the relationship between the explanatory variable and the target variable, which can then be used to predict the responses of new explanatory variables that were not part of the dataset. Linear regression can be understood as finding the best fitting straight line through the training examples. This line is also called the **regression line**, and the vertical lines from the regression line to the training examples are the so called **offsets** or **residuals**, the errors of our prediction.
- We can generalize the linear regression model to multiple explanatory variables, this process is called **multiple linear regression**. Visualization of multiple linear regression hyperplanes in a three-dimensional scatter plot are already challenging to interpret when looking at static figures.
- In contrast to common belief, training a linear regression model does not require that the explanatory variables are normally distributed. The normality assumption is only a requirement for certain statistics and hypothesis tests.
- The correlation matrix is a square matrix that contains the **Pearson product-moment correlation coefficient** (often abbreviated as Pearson's r) which measures the linear dependence between pairs of features. The correlation coefficients are in the range -1 to 1. Two features have a perfect positive correlation if r = 1, no correlation if r = 0, and a perfect negative correlation if r = -1. As mentioned previously, Pearson's correlation coefficient can simply be calculated as the covariance between two features, x and y (numerator), divided by the product of their standard deviations (denominator).
- Linear regression models can be heavily impacted by the presence of outliers. In certain situations, a very small subset of our data can have a big effect on the estimated model coefficients. Many statistical tests can be used to detect outliers. However, removing outliers always requires our own judgment as data scientists as well as our domain knowledge.
- As an alternative to throwing out outliers, we will look at a robust method of regression using the **RANdom SAmple Consensus (RANSAC)** algorithm, which fits a regression model to a subset of the data, the so-called inliers.
- We can summarize the RANSAC algorithm as follows:
  - Select a random number of examples to be inliers and fit the model.
  - Test all other data points against the fitted model and add those points that fall within a user given tolerance to the inliers.
  - Refit the model using all inliers.
  - Estimate the error of the fitted model versus the inliers.
  - Terminate the algorithm if the performance meets a certain user-defined threshold or if a fixed number of iterations was reached, go back to step 1 otherwise.
- **Residual plots** are a commonly used graphical tool for diagnosing regression models. They can help to detect nonlinearity and outliers and check whether the errors are randomly distributed.
- In the case of a perfect prediction, the residuals would be exactly zero, which we will probably never encounter in realistic and practical applications. However, for a good regression model, we would expect the errors to be randomly distributed and the residuals to be randomly scattered around the centerline. If we see patterns in a residual plot, it means that our model is unable to capture some explanatory information, which has leaked into the residuals, as you can see in our previous residual plot. Furthermore, we can use residual plots to detect outliers, which are represented by the points with a large deviation from the centerline.
- Another useful quantitative measure of a model's peformance is the **mean squared error (MSE)** that is the loss function that we minimize to fit the linear regression model. Similar to prediction accuracy in classification contexts, we can use the MSE for cross-validation and model selection. Like classification accuracy, MSE also normalizes according to the sample size. This makes it possible to compare across different sample sizes as well.
- Note that it can be more intuitive to show the error on the original unit scale, which is why we may choose to compute the square root of MSE, called root mean squared error, or the **mean absolute error (MAE)**, which emphasizes incorrect prediction slightly less.
- When we use the MAE or MSE for comparing models, we need to be aware that these are unbounded in contrast to the classification accuracy, for example. In other words, the interpretations of the MAE and MSE depend on the dataset and feature scaling. For example, if the sale prices were presented as multiples of 1000, the same model would yield a lower MAE compared to a model that worked with unscaled features. Thus, it may sometimes be more useful to report the **coefficient of determination ($R^2$)**, which can be understood as a standardized version of the MSE, for better interpretability of the model's performance. Or, in other words, $R^2$ is the fraction of response variance that is captured by the model.
- SSE is the sum of squared errors, which is similar to MSE but does not include the normalization by sample size n.
- SST is the total sum of squares, which is the variance of the response.
- For the training dataset, $R^2$ is bounded between 0 and 1, but it can become negative for the test dataset. A negative $R^2$ means that the regression model fits the data worse than a horizontal line representing the sample mean. In practice, this often happens in the case of extreme overfitting, or if we forget to scale the test set in the same manner we scaled the training set. If $R^2 = 1$, the model fits the data perfectly with a corresponding $MSE = 0$.
- Regularization is one approach to tackling the problem of overfitting by adding additional information and thereby shrinking the parameter values of the model to induce a penalty against complexity. The most popular approaches to regularized linear regression are the so-called **ridge regression, least absolute shrinkage and selection operator (LASSO)** and **elastic net**.
- Ridge regression is an L2-penalized model where we simply add the squared sum of the weights to the MSE loss function. By increasing the value of the hyperparameter $\lambda$, we increase the regularization strength and thereby shrink the weights of our model.
- An alternative approach that can lead to sparse models is LASSO. Depending on the regularization strength, certain weights can become zero, which also makes LASSO useful as a supervised feature selection technique. The L1 penalty for LASSO is defined as the sum of the absolute magnitudes of the model weights.
- A limitation of LASSO is that it selects at most n features if $m > n$, where n is the number of training examples. This may be undesirable in certain applications of feature selection. In practice, however, this property of LASSO is often an advantage because it avoids saturated models. The saturation of a model occurs if the number of training examples is equal to the number of features, which is a form of overparameterization. As a consequence, a saturated model can always fit the training data perfectly but is merely a form of interpolation and thus is not expected to generalize well.
- A compromise between ridge regression and LASSO is elastic net, which has an L1 penalty to generate sparsity and an L2 penalty such that it can be used for selecting more than n features if $m > n$.
- One way to account for the violation of the linearity assumption between explanatory and response variables is to use a polynomial regression model by adding polynomial terms. Although we can use polynomial regression to model a nonlinear relationship, it is still considered a multiple linear regression model because of the linear regression coefficients.
- An advantage of the decision tree algorithm is that it works with arbitrary features and does not require any transformation of the features if we are dealing with nonlinear data because decision trees analyze one feature at a time, rather than taking weighted combinations into account. (Likewise, normalizing or standardizing is not required for decision trees.)
- When we used decision trees for classification, we defined entropy as a measure of impurity to determine which feature split maximizes the information gain. Here, $x_i$ is the feature to perform the split, $N_p$ is the number of training examples at the parent node, and $D_{left}$ and $D_{right}$ are the subsets of the training examples at the left and right child nodes after the split. To use a decision tree for regression, we need an impurity metric that is suitable for continous variables, so we define the impurity measure of a node as the MSE instead. In the context of decision tree regression, the MSE is often referred to as **within-node variance**, which is why the splitting criterion is also better known as **variance reduction**.
- A random forest usually has a better generalization performance than an individual tree due to randomness, which helps to decrease the model's variance. Other advantages of random forests are that they are less sensitive to outliers in the dataset and don't require much parameter tuning. The only parameter in random forests that we typically need to experiment with is the number of trees in the ensemble. The difference between random forest algorithm for regression and classification is that we use the MSE criterion to grow the individual decision trees, and the predicted target variable is calculated as the average prediction across all decision trees.

## Key Terms/Formulas

Linear model with one explanatory variable:

$$
y = w_1x + b
$$

Multiple linear regression:

$$
y = w_1 x_1 + \ldots + w_m x_m + b = \sum_{i=1}^{m}w_i x_i + b = w^{T}x + b
$$

Pearson's coefficient:

$$
r = \frac{\sum_{i=1}^{n}\left[(x^{(i)}-\mu_x)(y^{(i)}-\mu_y)\right]}
{\sqrt{\sum_{i=1}^{n}(x^{(i)}-\mu_x)^2}
\sqrt{\sum_{i=1}^{n}(y^{(i)}-\mu_y)^2}}
= \frac{\sigma_{xy}}{\sigma_x \sigma_y}
$$

Mean squared error:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n}(y^{(i)} - \hat{y}^{(i)})^2
$$

Mean absolute error:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n}|(y^{(i)} - \hat{y}^{i})|
$$

Coefficient of determination:

$$
R^2 = 1 - \frac{SSE}{SST}
$$

Sum of squared errors:

$$
SSE = \sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2
$$

Total sum of squares:

$$
SST = \sum_{i=1}^n (y^{(i)} - \mu_y)^2
$$

Ridge regression loss function:

$$
L(w)_{Ridge} = \sum_{i=1}^{n}(y^{(i)} - \hat{y}^{(i)})^2 + \lambda \lVert \mathbf{w} \rVert _2^{2}
$$

Lasso loss function:

$$
L(w)_{Lasso} = \sum_{i=1}^{n}(y^{(i)} - \hat{y}^{(i)})^2 + \lambda \lVert \mathbf{w} \rVert _1
$$

Elastic net loss function:

$$
L(w)_{Elastic Net} = \sum_{i=1}^{n}(y^{(i)} - \hat{y}^{(i)})^2 + \lambda _2 \lVert \mathbf{w} \rVert _2^{2} + \lambda _1 \lVert \mathbf{w} \rVert _1
$$

Information Gain:

$$
IG (D_p, x_i) = I(D_p) - \frac{N_{left}}{N_p}I(D_{left}) - \frac{N_{right}}{N_p}I(D_{right})
$$
