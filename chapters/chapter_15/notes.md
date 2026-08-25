# Chapter 15 - Modeling Sequential Data Using Recurrent Neural Networks

**Status:** Completed
**Code:** Neural Networks  
**Focus:** Introducing sequential data, RNNs for modeling sequences, long short-term memory, truncated backpropagation through time, implementing a multilayer RNN for sequence modeling in PyTorch, RNN sentiment analysis of the IMDb movie review dataset, RNN character-level language modeling with LSTM cells, using text data from Jules Verne's The Mysterious Island, using gradient clipping to avoid exploding gradients

## Summary

- Let's begin our discussion of **recurrent neural networks (RNNs)** by looking at the nature of sequential data, which is more commonly known as sequence data or sequences. What makes sequences unique, compared to other types of data, is that elements in a sequence appear in a certain order and are not independent of each other. Typical machine learning algorithms for supervised learning assume that the input is **independent and identically distributed (IID)** data, which means that the training examples are mutually independent and have the same underlying distribution. In this regard, based on the mutual independence assumption, the order in which the training examples are given to the model is irrelevant.For example, if we have a sample consisting of n training examples, $x^{(1)}, x^{(2)}, \cdots, x^{(n)}$, the order in which we use the data for training our machine learning algorithm does not matter. An example of this scenario would be the Iris dataset that we worked with previously. In the Iris dataset, each flower has been measured independently, and the measurements of one flower do not influence the measurements of another flower.
- However, this assumption is not valid when we deal with sequences - by definition, order matters. Predicting the market value for a particular stock would be an example of this scenario. For instance, assume we have a sample of $n$ training examples, where each training example represents the market value of a certain stock on a particular day. If our task is to predict the stock market value for the next three days, it would make sense to consider the previous stock prices in a date-sorted order to derive trends rather than utilize these training examples in a randomized order.
- Time series data is a special type of sequential data where each example is associated with a dimension for time. In time series data, samples are taken at successive timestamps, and therefore, the time dimension determines the order among the data points. For example, stock prices and voice or speech records are time series data.
- On the other hand, not all sequential data has the time dimension. For example, in text data or DNA sequences, the examples are ordered, but text or DNA does not qualify as time series data.
- We have established that order among data points is important in sequential data, so we next need to find a way to leverage this ordering information in a machine learning model. Throughout this chapter, we will represent sequences as $\langle x^{(1)}, x^{(2)}, \cdots, x^{(T)} \rangle$. The superscript indices indicate the order of the instances, and the length of the sequence is $T$. For a sensible example of sequences, consider time series data, where each example point, $x^{(t)}$, belongs to a particular time, $t$.
- As we have already mentioned, the standard NN models that we have covered so far, such as **multilayer perceptrons (MLPs)** and CNNs for image data, assume that the training examples are independent of each other and thus do not incorporate ordering information. We can say that such models do not have a memory of previously seen training examples. For instance, the samples are passed through the feedforward and backpropagation steps, and the weights are updated independently of the order in which the training examples are processed.
- RNNs, by contrast, are designed for modeling sequences and are capable of remembering past information and processing new events accordingly, which is a clear advantage when working with sequences data.
- Sequence modeling has many fascinating applications, such as language translation, image captioning, and text generation. However, in order to choose an appropriate architecture and approach, we have to understand and be able to distinguish between these different sequence modeling tasks. Let's discuss the different relationship categories between input and output data. If neither the input nor output data represent sequences, then we are dealing with standard data, and we could simply use a multilayer perceptron to model such data. However, if either the input or output is a sequence, the modeling task likely falls into one of these categories:
  - **Many-to-one**: The input data is a sequence, but the output is a fixed-size vector or scalar, not a sequence. For example, in sentiment analysis, the input is text-based (for example, a movie review) and the output is a class label (for example, a label denoting whether a reviewer liked the movie).
  - **One-to-many**: The input data is in standard format and not a sequence, but the output is a sequence. An example of this category is image captioning - the input is an image and the output is an English phrase summarizing the content of that image.
  - **Many-to-many**: Both the input and output arrays are sequences. This category can be further divided based on whether the input and output are synchronized. An example of a synchronized many-to-many modeling task is video classification, where each frame in a video is labeled. An example of a delayed many-to-many modeling task would be translating one laneguage into another. For instance, an entire English sentence must be read and processed by a machine before its translation into German is produced.
- This generic RNN architecture could correspond to the two sequence modeling categories where the input is a sequence. Typically, a recurrent layer can return a sequence as output, $\langle o^{(0)}, o^{(1)}, \cdots, o^{(T)} \rangle$, or simply return the last output (at $t=T$, that is, $o^{(T)}$). Thus, it could be either many-to-many, or it could be many-to-one if, for example, we only use the last element, $o^{(T)}$, as the final output.
- In a standard feedforward network, information flows from the input to the hidden layer, and then from the hidden layer to the output layer. On the other hand, in an RNN, the hidden layer receives its input from both the input layer of the current time step and the hidden layer from the previous time step.
- The flow of information in adjacent time steps in the hidden layer allows the network to have a memory of past events. This flow of information is usually displayed as a loop, also known as a **recurrent edge** in graph notation, which is how this general RNN architecture got its name.
- Similar to multilayer perceptrons, RNNs can consist of multiple hidden layers. Note that it's a common convention to refer to RNNs with one hidden layer as a single-layer RNN, which is not to be confused with single-layer NNs without a hidden layer, such as Adaline or logistic regression.
- As we know, each hidden unit in a standard NN receives only one input - the net preactivation associated with the input layer. In contrast, each hidden unit in an RNN receives two distinct sets of input - the preactivation from the input layer and the activation of the same hidden layer from the previous time step, $t - 1$.
- At the first step, $t=0$, the hidden units are initialized to zeros or small random values. Then, at a time step where $t > 0$, the hidden units receive their input from the data point at the current time, $x^{(t)}$, and the previous values of hidden units at $t - 1$, indicated as $h^{(t-1)}$.
- Similarly, in the case of a multilayer RNN, we can summarize the information flow as follows:
  - layer = 1: Here, the hidden layer is represented as $h_{1}^{(t)}$ and it receives its inputs from the data point, $x^{(t)}$, and the hidden values in the same layer, but at the previous time step, $h_{1}^{(t-1)}$.
  - layer = 2: The second hidden layer, $h_{2}^{(t)}$, receives its inputs from the outputs of the layer below at the current time step $(o_{1}^{(t)})$ and its own hidden values from the previous time step, $h_{2}^{(t-1)}$.
- Since, in this case, each recurrent layer must receive a sequence as input, all the recurrent layers except the last one must return a sequence as output (that is, we will later have to set return_sequences=True). The behavior of the last recurrent layer depends on the type of problem.
- Now that you understand the structure and the general flow of information in an RNN, let's get more specific and compute the actual activations of the hidden layers, as well as the output layer. For simplicity, we will consider just a single hidden layer, however, the same concept applies to multilayer RNNs.
- Each directed edge (the connection between boxes) in the representation of an RNN is associated with a weight matrix. Those weights do not depend on time, $t$; therefore, they are shared across the time axis. The different weight matrices in a single-layer RNN are as follows:
  - $W_{xh}$: The weight matrix between the input, $x^{(t)}$, and the hidden layer, h
  - $W_{hh}$: The weight matrix associated with the recurrent edge
  - $W_{ho}$: The weight matrix between the hidden layer and the output layer
- In certain implementations, you may observe that the weight matrices $W_{xh}$ and $W_{hh}$, are concatenated to a combined matrix, $W_h = [W_{xh};W_{hh}]$. Later in this section, we will make use of this notation as well.
- Computing the activations is very similar to standard multilayer perceptrons and other types of feedforward NNs. For the hidden layer, the net input, $z_h$ (preactivation), is computed through a linear combination; that is, we compute the sum of the multiplications of the weight matrices with the corresponding vectors and add the bias unit (eq 1). The activation of the hidden units as the time step $t$ are calculated after (eq 2). Once the activations of the hidden units at the current time step are computed, then the activations of the output units will be computed (eq 3).
- The learning algorithm for RNNs is called **backpropagation through time (BPTT)**. The derivation of the gradients might be a bit complicated, but the basic idea is that the overall loss, L, is the sum of all the loss functions at times $t = 1$ to $t = T$ (eq 4).
- Since the loss at time $t$ is dependent on the hidden units at all previous time steps $1:t$, the gradient will be computed as follows (eq 5).
- So far, we have seen recurrent networks in which the hidden layer has the recurrent property. However, note that there is an alternative model in which the recurrent connection comes from the output layer. In this case, the net activations from the output layer at the previous time step, $o^{(t-1)}$, can be added in one of two ways:
  - To the hidden layer at the current time step, $h^t$ (output-to-hidden recurrence)
  - To the output layer at the current time step, $o^t$ (output-to-output recurrence)
- The differences between these architectures can be clearly seen in the recurring connections. Following our notation, the weights associated with the recurrent connection will be denoted for the hidden-to-hidden recurrence by $W_{hh}$, for the output-to-hidden recurrence by $W_{oh}$, and for the output-to-output recurrence by $W_{oo}$. In some articles in literature, the weights associated with the recurrent connections are also denoted by $W_{rec}$.
- BPTT, which was briefly mentioned earlier, introduces some new challenges. Because of the multiplicative factor, $\frac{\partial h^{(t)}}{\partial h^{(k)}}$, in computing the gradients of a loss function, the so-called **vanishing** and **exploding** gradient problems arise.
- Basically, $\frac{\partial h^{(t)}}{\partial h^{(k)}}$ has $t-k$ multiplications; therefore, multiplying the weight, $w$ by itself $t-k$ times the results in a factor, $w^{t-k}$. As a result, if $\lvert w \rvert < 1$, this factor becomes very small when $t-k$ is large. On the other hand, if the weight of the recurrent edge is $\lvert w \rvert > 1$, then $w^{t-k}$ becomes very large when $t-k$ is large. Note that a large $t-k$ refers to long-range dependencies. We can see that the naive solution to avoid vanishing or exploding gradients can be reached by ensuring $\lvert w \rvert = 1$. In practice, there are at least three solutions to this problem:
  - Gradient clipping
  - Truncated backpropagation through time (TBPTT)
  - LSTM
- Using gradient clipping, we specify a cut-off or threshold value for the gradients, and we assign this cut-off value to the gradient values that exceed this value. In contrast, TBPTT simply limits the number of time steps that the signal can backpropagate after each forward pass. For example, even if the sequence has 100 elements or steps, we may only backpropagate the most recent 20 time steps.
- While both gradient clipping and TBPTT can solve the exploding gradient problem, the truncation limits the number of steps that the gradient can effectively flow back and properly update the weights. On the other hand, LSTM, has been more successful in vanishing and exploding gradient problems while modeling long-range dependencies through the use of memory cells.
- As stated previously, LSTMs were first introduced to overcome the vanishing gradient problem. The building block of an LSTM is a **memory cell**, which essentially represents or replaces the hidden layer of standard RNNs.
- In each memory cell, there is a recurrent edge that has the desirable weight, $w=1$, as we discussed, to overcome the vanishing and exploding gradient problems. The values associated with this recurrent edge are collectively called the **cell state**.
- Notice that the cell state from the previous time step, $C^{(t-1)}$, is modified to get the cell state at the current time step, $C^{(t)}$, without being multiplied directly by any weight factor. The flow of information in this memory cell is controlled by several computation units (often called gates) that will be described here. Four boxes are indicated with an activation function, either the sigmoid function or tanh, and a set of weights; these boxes apply a linear combination by performing matrix-vector multiplications on their inputs (which are $h^{(t-1)}$ and $x^{(t)}$). These units of computation with sigmoid activation functions, whose output units are passed through, are called gates.
- In an LSTM cell, there are three different types of gates, which are known as the forget gate, the input gate, and the output gate:
  - The **forget gate ($f_t$)** allows the memory cell to reset the cell state without growing indefinitely. In fact, the forget gate decides which information is allowed to go through and which information to suppress. Now, $f_t$, is computed as follows (eq 6). Note that the forget gate was not part of the original LSTM cell, it was added a few years later to improve the original model.
  - The **input gate ($i_t$)**: and **candidate value ($\widetilde{C}_t$)** are responsible for updating the cell state. They are computed as follows (eq 7).
  - The cell state at time $t$ is computed as follows (eq 8)
  - The **output gate ($o_t$) decides how to update the values of hidden units (eq 9)
  - The hidden units at the current time step are computed as follows (eq 10)
- LSTMs provide a basic approach for modeling long-range dependencies in sequences. Yet, it is important to note that there are many variations of LSTMs described in literature. Also worth noting is a more recent approach, **gated recurrent unit (GRU)**, which was proposed in 2014. GRUs have a simpler architecture than LSTMs; therefore, they are computationally more efficient, while their performance in some tasks, such as polyphonic music modeling, is cmomparable to LSTMs.
  
### Key Terms/Formulas

Hidden layer preactivation for RNN:

$$
z_{h}^{(t)} = W_{xh}x^{(t)} + W_{hh}h^{(t-1)} + b_h
$$

Activation of the hidden unit:

$$
h^{(t)} = \sigma_{h}(z_{h}^{(t)}) = \sigma_{h}(W_{xh}x^{(t)} + W_{hh}h^{(t-1)} + b_h)
$$

Activation of the output unit:

$$
o^{(t)} = \sigma_0 (W_{ho}h^{(t)} + b_0)
$$

Loss function in RNN:

$$
L = \sum_{t=1}^{T} L^{(t)}
$$

Gradient of loss function in RNN:

$$
\frac{\partial L^{(t)}}{\partial W_{hh}} = \frac{\partial L^{(t)}}{\partial o^{(t)}} \times \frac{\partial o^{(t)}}{\partial h^{(t)}} \times (\sum_{k=1}^t \frac{\partial h^{(t)}}{\partial h^{(k)}} \times \frac{\partial h^{(k)}}{\partial W_{hh}})
$$

Forget gate (LSTM):

$$
f_t = \sigma(W_{xf}x^{(t)} + W_{hf}h^{(t-1)} + b_f)
$$

Input gate and the candidate value (LSTM):

$$
\begin{aligned}
\boldsymbol{i}_t
  &= \sigma\left(\boldsymbol{W}_{xi}\boldsymbol{x}^{(t)} + \boldsymbol{W}_{hi}\boldsymbol{h}^{(t-1)} + \boldsymbol{b}_i\right) \\
\widetilde{\boldsymbol{C}}_t &= \tanh\left(\boldsymbol{W}_{xc}\boldsymbol{x}^{(t)} + \boldsymbol{W}_{hc}\boldsymbol{h}^{(t-1)} + \boldsymbol{b}_c\right)
\end{aligned}
$$

Cell state at time t (LSTM):

$$
\boldsymbol{C}^{(t)} = \left(\boldsymbol{C}^{(t-1)} \odot \boldsymbol{f}_t\right) \oplus \left(\boldsymbol{i}_t \odot \widetilde{\boldsymbol{C}}_t\right)
$$

Output gate (LSTM):

$$
o_t = \sigma(W_{xo}x^{(t)} + W_{ho}h^{(t-1)} + b_o)
$$

Hidden units (LSTM):

$$
h^{(t)} = o_t \odot \tanh(C^{(t)})
$$