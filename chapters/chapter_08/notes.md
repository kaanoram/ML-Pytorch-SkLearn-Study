# Chapter 8 - Applying Machine Learning to Sentiment Analysis

**Status:** Completed  
**Code:** Sentiment Analysis  
**Focus:** Cleaning and preparing text data, building feature vectors from text documents, training a machine learning model to classify positive and negative movie reviews, working with large text datasets using out-of-core learning, inferring topics from document collections for categorization

## Summary  

- **Sentiment analysis (opinion mining)**, is a popular subdiscipline of the broader field of NLP. It is concerned with analyzing the sentiment of documents. A popular task in sentiment analysis is the classification of documents based on the expressed opinions or emotions of the authors with regards to a particular topic.
- The **bag-of-words** model allows us to represent text as numerical feature vectors. The idea behind it is:
  - We create a vocabulary of unique tokens, for example, words from the entire set of documents.
  - We construct a feature vector from each document that contains the counts of how often each word occurs in the particular document.
- Since the unique words in each document represent only a small subset of all the words in the bag-of-words vocabulary, the feature vectors will consist mostly of zeros, which is why we call them **sparse**.
- The values in the feature vectors are also called **raw term frequencies** tf(t, d), the number of times a term t occurs in document d. It should be noted that in the bag-of-words model, the word or term order in a sentence or document does not matter. The order in which the term frequencies appear in the feature vector is derived from the vocabulary indices, which are usually assigned alphabetically.
- The sequence of items in bag-of-words model that we just created is also called 1-gram or unigram model, each item or token in the vocabulary represents a single word. More generally, the contiguous sequences of items in NLP - words, letters, or symbols - are also called **n-grams**. The choice of the number n in the n-gram model depends on the particular application. For example, n-grams of size 3 and 4 yield good performances in the anti-spam filtering of email messages.
- When we are analyzing text data, we often encounter words that occur across multiple documents from both classes. These frequently occurring words typically don't contain useful or discriminatory information. A technique called **term frequency-inverse document frequency (tf-idf)** can be used to downweight these frequently occurring words in feature vectors. The tf-idf can be definted as the product of the term frequency and the inverse document frequency: $tf-idf(t, d) = tf(t, d) \times idf(t, d)$
- In the context of tokenization, a useful technique is **word stemming**, which is the process of transforming a word into its root form. The original stemming algorithm is also known as **Porter stemmer algorithm**. It allows us to map related words to the same stem. The Natural Language Toolkit (NLTK) for Python implements the Porter stemming algorithm.
- The Porter stemming algorithm is probably the oldest and simplest stemming algorithm. Other popular stemming algorithms include the newer Snowball stemmer (Porter2 or English stemmer) and the Lancaster stemmer (Paice/Husk stemmer). While both the Snowball and Lancaster stemmers are faster than the original Porter stemmer, the Lancaster stemmer is also notorious for being more aggressive than the Porter stemmer, which means that it will produce shorter and more obscure words. These alternative stemming algorithms are also available through the NLTK package.
- While stemming can create non-real words, such as 'thu' (from 'thus'), a technique called lemmatization aims to obtain the canonical forms of individual words (lemmas). However, lemmatization is computationally more difficult and expensive compared to stemming and in practice, it has been observed that stemming and lemmatization have little impact on the performance of text classification.
- Another important topic is **stop word removal**. Stop words (e.g.is, and, has, like) are simply those words that are extremely common in all sorts of texts and probably bear no useful information that can be used to distinguish between classes of documents. Removing stop words can be useful if we are working with raw or normalized term frequencies rather than tf-idfs, which already downweight the frequently occurring words.
- A still very popular classifier for text classification is **Naive Bayes classifier**, which gained popularity in applications of email spam filtering. Naive Bayes classifiers are easy to implement, computationally efficient, and tend to perform particularly well on relatively small datasets compared to other algorithms.
- A more modern alternative to the bag-of-words model is **word2vec**. The word2vec algorithm is an unsupervised learning algorithm based on neural networks that attempt to automatically learn the relationship between words. The idea behind word2vec is to put words that have similar meanings into similar clusters, and via clever vector spacing, the model can reproduce certain words using simple vector math.
- **Topic modeling** describes the broad task of assigning topics to unlabeled text documents. For example, a typical application is the categorization of documents in a large text corpus of newspaper articles. In applications of topic modeling, we then aim to assign category labels to those articles, for example, sports, finance, world news, politics, and local news. It is a clustering task, a subcategory of unsupervised learning.
- **Latent Dirichlet allocation (LDA)** is a generative probabilistic model that tries to find groups of words that appear frequently together across different documents. These frequently appearing words represent our topics, assuming that each document is a mixture of different words. The input to an LDA is the bag-of-words model.
- Given a bag-of-words matrix as input, LDA decomposes it into two new matrices: a document-to-topic matrix and a word-to-topic matrix
- LDA decomposes the bag-of-words matrix in such a way that if we multiply those two matrices together, we will be able to reproduce the input, the bag-of-words matrix, with the lowest possible error. In practice, we are interested in those topics that LDA found in the bag-of-words matrix. The only downside may be that we must define the number of topics beforehand - the number of topics is a hyperparameter of LDA that has to be specified manually.

## Key Terms/Formulas

Inverse document frequency:

$$
idf(t, d) = log\frac{n_d}{1 + df(d, t)}
$$
