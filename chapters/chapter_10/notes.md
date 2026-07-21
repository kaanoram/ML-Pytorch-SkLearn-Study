# Chapter 10 - Working with Unlabeled Data - Clustering Analysis

**Status:** Completed  
**Code:** Clustering Analysis  
**Focus:** Finding centers of similarity using the popular k-means algorithm, taking a bottom-up approach to building hierarchical clustering trees, identifying arbitrary shapes of objects using a density-based clustering approach  

## Summary

- The goal of **clustering** is to find a natural grouping in data so that items in the same cluster are more similar to each other than those from different clusters. Examples of business-oriented applications of clustering include grouping of documents, music, and movies by different topics, or finding customers that share similar interests based on common purchase behaviors as a basis for recommendation engines.
- **k-means** algorithm is extremely easy to implement, but it is also computationally very efficient compared to the other clustering algorithms, which might explain its popularity. The k-means algorithm belongs to the category of **prototype-based clustering**.
- Prototype-based clustering means that each cluster is represented by a prototype, which is usually either a **centroid** of similar points with continuous features, or the **medoid** (the most representative or the point that minimizes the distance to all other points that belong to a particular cluster) in the case of categorical features. While k-means is very good at identifying clusters with a spherical shape, one of the drawbacks of this clustering algorithm is that we have to specify the number of clusters, k, a priori. An inappropriate choice for k can result in poor clustering performance. **Elbow** method and **silhouette plots** are useful techniques to evaluate the quality of a clustering to help us determine the optimal number of clusters, k.
- In real-world applications of clustering, we do not have any ground-truth category information (information provided as empirical evidence as opposed to inference) about those examples, if we were given class labels, this task would fall into the category of supervised learning. Thus, our goal is to group the examples based on their feature similarities, which can be achieved using the k-means algorithm:
  - Randomly pick k centroids from the examples as initial cluster centers
  - Assign each example to the nearest centroid, $\mu^(j)$,
  - Move the centroid to the center of the examples that were assigned to it
  - Repeat steps 2 and 3 until the cluster assignments do not change or a user-defined tolerance or maximum number of iterations is reached
- We can define similarity as the opposite of distance, and a commonly used distance metric for clustering examples with continuous features is the **squared Euclidean distance** between two points, x, and y, in m-dimensional space.
- Based on this Euclidean distance metric, we can describe the k-means algorithm as a simple optimization problem, an iterative approach for minimizing the within-cluster **sum of squared errors (SSE)**, which is sometimes also called **cluster inertia**. Here, $\mu ^(j)$ is the representative point (centroid) for cluster j. $w^{(i, j)} = 1$ if the example, $x^(i)$ is in cluster j, or 0 otherwise.
- A problem with k-means is that one or more clusters can be empty. Note that this problem does not exist for k-medoids or fuzzy C-means. However, this problem is accounted for in the current k-means implementation in scikit-learn. If a cluster is empty, the algorithm will search for the example that is farthest away from the centroid of the empty cluster. Then, it will reassign the centroid to be this farthest point.
- When we are applying k-means to real-world data using a Euclidean distance metric, we want to make sure that the features are measured on the same scale and apply z-score standardization or min-max scaling if necessary.
- So far, we have discussed the classic k-means algorithm, which uses a random seed to place the initial centroids, which can sometimes result in bad clusterings or slow convergence if the initial centroids are chosen poorly. One way to address this issue is to run the k-means algorithm multiple times on a dataset and choose the best-performing model in terms of the SSE. Another strategy is to place the initial centroids far away from  each other via the k-means++ algorithm which leads to better and more consistent results than the classic k-means.
- The initialization in k-means++ can be summarized as follows:
  - Initialize an empty set, M, to store the k centroids being selected
  - Randomly choose the first centroid $\mu^{(j)}$ from the input examples and assign it to M.
  - For each example, $x^{(i)}$, that is not in M, find the minimum squared distance, $d(x^{(i)}, M)^2, to any of the centroids in M.
  - To randomly select the next centroid, $\mu^{(p)}$, use a weighted probability distribution equal to $\frac{d(\mu^{(p)}, M)^2}{\sum_{i}d(x^{(i)}, M)^2}$. For instance, we collect all points in an array and choose a weighted random sampling, such that the larger the squared distance, the more likely a point gets chosen as the centroid.
  - Repeat steps 3 and 4 until k centroids are chosen.
  - Proceed with the classic k-means algorithm.
- **Hard clustering** describes a family of algorithms where each example in a dataset is assigned to exactly one cluster, as in the k-means and k-means++ algorithms that we discussed earlier in this chapter. In contrast, algorithms for **soft clustering** (sometimes also called **fuzzy clustering**) assign an example to one or more clusters. A popular example of soft clustering is the **fuzzy C-means (FCM)** algorithm (also called **soft k-means** or **fuzzy k-means**).
- The FCM procedure is very similar to k-means. However, we replace the hard cluster assignment with probabilities for each point belonging to each cluster. In k-means we could express the cluster membership of an example x with a sparse vector of binary values. In contrast, a membership vector in FCM would have values that fall in range [0, 1] representing the probability of membership of the respective cluster centroid. The sum of the memberships for a given example is equal ot 1. As with the k-means algorithm, we can summarize the FCM algorithm in four key steps:
  - Specify the number of k centroids and randomly assign the cluster memberships for each point
  - Compute the cluster centroids $\mu^{(j)}$
  - Update the cluster memberships for each point
  - Repeat steps 2 and 3 until the membership coefficients do not change or a user defined tolerance or maximum number of iterations is reached.
- Just by looking at the equation to calculate the cluster memberships, we can say that each iteration in FCM is more expensive than an iteration in k-means. On the other hand, FCM typically requires fewer iterations overall to reach convergence. However, it has been found, in practice, that both k-means and FCM produce very similar clustering outputs.
- Another intrinsic metric to evaluate the quality of clustering is **silhouette analysis**, which can also be applied to clustering algorithms other than k-means. Silhouette analysis can be used as a graphical tool to plot a measure of how tightly grouped examples in the cluster are. To calculate the **silhouette coefficient** of a single example in our dataset, we can apply the following steps:
  - Calculate the **cluster cohesion**, $a^{(i)}$ as the average distance between an example $x^{(i)}$ and all other points in the same cluster
  - Calculater **cluster separation**, $b^{(i)}$ from the next closest cluster as the average distance between the example $x^{(i)}$ and all examples in the nearest cluster.
  - Calculate the silhouette $s^{(i)}$ as the difference between cluster cohesion and separation divided by the greater of the two.
- The silhouette coefficient is bounded in the range -1 to 1. We can see that the silhouette coefficient is 0 if the cluster separation and cohesion are equal. Furthermore, we get close to an ideal silhouette coefficient of 1 if $b^{(i)} >> a^{(i)}$, since $b^{(i)}$ quantifies how dissimilar an example is from other clusters, and $a^{(i)}$ tells us how similar it is to the other examples in its own cluster.
- An alternative approach to prototype based clustering is **hierarchical clustering**. One advantage of hierarchical clustering algorithm is that it allows us to plot **dendrograms** (visualization of a binary hierarchical clustering), which can help with the interpretation of the results by creating meaningful taxonomies. Another advantage of this hierarchical approach is that we do not need to specify the number of clusters upfront.
- The two main approaches to hierarchical clustering are **agglomerative** and **divisive** hierarchical clustering. In divisive hierarchical clustering, we start with one cluster that encompasses the complete dataset, and we iteratively split the cluster into smaller clusters until each cluster only contains one example. Agglomerative clustering is the opposite approach - each example is an individual cluster and the closest pairs are merged until only one cluster remains.
- The two standard algorithms for agglomerative hierarchical clustering are **single linkage** and **complete linkage**. Using single linkage, we compute the distances between the most similar members for each pair of clusters and merge the two clusters for which the distance between the most similar members is the smallest. The complete linkage approach is similar to single linkage but, instead of comparing the most similar members in each pair of clusters, we compare the most dissimilar members to perform the merge.
- Other commonly used algorithms for agglomerative hierarchical clustering include average linkage and Ward's linkage. In average linkage, we merge the cluster pairs based on the minimum average of the distances between all group members in the two clusters. In Ward's linkage, the two clusters that lead to the minimum increase of the total within-cluster SSE are merged.
- Hierarchical complete linkage clustering is an iterative procedure that can be summarized by the following steps:
  - Compute a pari-wise distance matrix of all examples
  - Represent each data point as a singleton cluster.
  - Merge the two closest clusters based on the distance between the most dissimilar (distant) members.
  - Update the cluster linkage matrix.
  - Repeat steps 2-4 until one single cluster remains.
- Another approach to clustering is called **density-based spatial clustering of applications with noise (DBSCAN)**, which does not make assumptions about spherical clusters like k-means, nor does it partition the dataset into hierarchies that require a manual cut-off point. As its name implies, density-based clustering assigns cluster labels based on dense regions of points. In DBSCAN, the notion of density is defined as the number of points within a specified radius.
- According to the DBSCAN algorithm, a special label is assigned to each example (data point) using the following criteria:
  - A point is considered a **core point** if at least a specified number of neighboring points (MinPts) fall within the specified radius, $\eps$
  - A **border point** is a point that has fewer neighbors than MinPts within $\epsilon$, but lies within the $\epsilon$ radius of a core point.
  - All other points that are neither core nor border points are considered **noise points**
- After labeling the points as core, border, or noise, the DBSCAN algorithm can be summarized in two simple steps:
  - Form a separate cluster for each core point or connected group of core points. (Core points are connected if they are no farther away than $\epsilon$)
  - Assign each border point to the cluster of its corresponding core point.
- One of the main advantages of using DBSCAN is that it does not assume that the clusters have a spherical shape as in k-means. Furthermore, DBSCAN is different from k-means and hierarchical clustering in that it doesn't necessarily assign each point to a cluster but is capable of removing noise points.
- However, we should also note some of the disadvantages of DBSCAN. With an increasing number of features in our dataset - assuming a fixed number of training examples - the negative effect of the **curse of dimensionality** increases. This is especially a problem if we are using the Euclidean distance metric. However, the problem of the curse of dimensionality is not unique to DBSCAN, it also affects other clustering algorithms that use the Euclidean distance metric, for example, k-means and hierarchical clustering algorithms. In addition, we have two hyperparameters in DBSCAN (MinPts and $\epsilon$) that need to be optimized to yield good clustering results. Finding a good combination of MinPts and $\epsilon$ can be problematic if the density differences in the dataset are relatively large.
- There is another class of more advanced clustering algorithms: graph-based clustering. Probably the most prominent members of the graph-based clustering family are the spectral clustering algorithms. Although there are many different implementations of spectral clustering, what they all have in common is that they use the eigenvectors of a similarity or distance matrix to derive the cluster relationships.
- Note that in practice, it is not always obvious which clustering algorithm will perform best on a given dataset, especially if the data comes in multiple dimensions that make it hard or impossible to visualize. Furthermore, it is important to emphasize that a successful clustering does not only depend on the algorithm and its hyperparameters, rather the choice of an appropriate distance metric and the use of domain knowledge can help to guide the experimental setup can be even more important.
- In the context of the curse of dimensionality, it is common practice to apply dimensionality reduction techniques prior to performing clustering. Such dimensionality reduction techniques for unsupervised datasets include principal component analysis and t-SNE. Also, it is particularly common to compress datasets down to two-dimensional subspaces, which allows us to visualize the clusters and assigned labels using two-dimensional scatterplots, which are particularly helpful for evaluating the results.

## Key Terms/Formulas

Squared Euclidean Distance:

$$
d(x, y)^2 = \sum_{j=1}^m (x_j - y_j)^2 = \lVert \mathbf{w - y} \rVert _2^{2}
$$

Sum of Squared Errors (SSE)/Cluster Inertia:

$$
SSE = \sum_{i=1}^n \sum_{j=1}^{k} w^{(i, j)} \lVert \mathbf{x^{(i)} - \mu^{(j)}} \rVert _2^{2}
$$

Silhouette:

$$
s^{(i)} = \frac{b^{(i)} - a^{(i)}}{\max{b^{(i)}, a^{(i)}}}
$$
