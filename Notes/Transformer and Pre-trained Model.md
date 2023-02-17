#### Pre-trained Model & Transformer

**Part 1**-**Pre-trained Model**

- **Distributed Word Embedding Model** - **Word2Vec, GloVe, Fasttext** 

  1. Compared with **One-Hot** Encoding, **Word Embedding** generated vectors have similarities between similar-meaning words,  **more characteristic**.

  2. **Word2Vec**: 

     Training methods:

     - CBOW(Continuous Bag-of-Words Model)

       Predicts the current word vector based on context.

     - Skip-gram

       Use the current word to predict context.

  3. These pre-trained models output a specific dimension vector with a given word, and the representation of sentences and articles is just based on the basic calculation of words(like average, sum), in addition, **Out of Vocabulary** cannot be well solved.

  

- **Contextual Word Embedding Model** - **GPT, BERT**

  1. In order to solve the polysemy of the word embedding context features, an encoder is usually used to extract word context features:

     To extract single-segment text features, it is usually to let tokens see information about tokens. Common feature extraction methods include CNN, RNN, and Attention.

     ![img](https://pic2.zhimg.com/v2-ca9e1ae46f7d6c83cc45ddaea5c472e9_r.jpg)

  2. Solve OOV(Out of Vocabulary):

     Reduce the size of Vocabulary, and describe the internal links of related tokens (such as pre-suffix, tense, comparative level, and superlative level of adjectives), subwords segmentation algorithm has gradually become the standard allocation of tokenizer in NLP tasks:

     - Word hashing(n-gram)

       For example, if good code is with trigrams, the result is [#go, goo, ood, od#].



**Part 2 - Transformer**

Like CNN, RNN, Transformer is a kind of **Feature Extractor** of NLP.

- Structure: 

  <img src="https://pic4.zhimg.com/80/v2-f6380627207ff4d1e72addfafeaff0bb_1440w.webp" alt="img" style="zoom:33%;" />

- Process:

  - **Step 1**: Get Input for each Word

    **X** represents the input of a word, **X = Word Embedding + Word location Embedding**

  - **Step 2**: Encoding block

    Input **Xn * d** matrix, and output of encoders is the same dimension

  - **Step 3**: Decoding block

    Masked the words after (i+1) and used translated (1~i) words to translate the (i+1) word

- Step 1: **Input of Transformer**

  - Word Embedding(WE): Word2Vec, GloVe, OR trained from the transformer

  - Position Embedding(PE): 

    <img src="https://pic3.zhimg.com/v2-8b442ffd03ea0f103e9acc37a1db910a_r.jpg" alt="img" style="zoom: 33%;" />

    **pos**: word position  **d**: the dimension of embedding  **2i**/**2i+1**: odds/even dimensions

    **X = WE + PE**

    

- Step2: **Self-Attention**

  - **Add & Norm**

    Add indicates that Residual Connection prevents network degradation.

    Norm indicates Layer Normalization, which is used to normalize the activation values at each layer.

  - **Self-Attention Structure**

    1. Calculated by **Q**(query), **K**(key), and **V**(value).

       Linear variable matrix **WQ**, **WK**, and **WV**.

       Q = X * WQ      K= X * WK      V = X * WV (num of rows in X, Q, K, V represent num of words) 

    2. Output calculation

       <img src="https://pic2.zhimg.com/80/v2-9699a37b96c2b62d22b312b5e1863acd_1440w.webp" alt="img" style="zoom:33%;" />

       In the formula, the inner product of each row vector of matrix Q and K is calculated. In order to prevent the **inner product** from being too large, it is divided by the square root of dk.

       Using **Softmax** to calculate the **attention coefficient of each word for other words**, Softmax in the formula is Softmax for each row of the matrix, that is, the sum of each row becomes 1.

       The **Softmax matrix** can be **multiplied by V** to get the final output Z.

       **Z1** represents the **attention coefficient** of word 1 versus all other words.

  - **Multi-Head Attention**

    Multi-Head Attention is formed by the **combination** of multiple Self-Attention.

    If heads = 8, use **different self-attentions** to generate 8 Z(s)

    **Concat** the 8 outputs and use a **Linear layer** to generate the final Z

    

- Step 3: **Encoder structure**

  X -> Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm 

  <img src="https://pic3.zhimg.com/80/v2-a4b35db50f882522ee52f61ddd411a5a_1440w.webp" alt="img" style="zoom:33%;" />

  - **Add** refers to **X+MultiHeadAttention(X)**, which is a **residual connection** usually used to solve the problem of multi-layer network training, allowing the network to **focus only on the part that is currently different.**
  - **Norm** refers to **Layer Normalization**, which is commonly used in RNN structures. Layer Normalization uses inputs from each layer of neurons to have **the same mean and variance**, as it **speeds up convergence**.
  - The **Feed Forward** layer is relatively simple and is a **two-tier fully connected layer**. The first layer has the activation function **Relu** and the second layer does not use the activation function.
  - X is the input, and the Feed Forward ends up with an output matrix that has dimensions consistent with X

  Encoder block receives an input matrix X and outputs a matrix. An Encoder can be formed by stacking multiple Encoder blocks.

  

- Step 4: **Decoder structure**

  - The first (Masked) Multi-Head Attention

    Mask Matrix: Word i can only use the information from 0 to i.

    After calculating the Q, K, V, and inner product QKT, **multiply with masked matrix** before softmax

    <img src="https://pic2.zhimg.com/v2-35d1c8eae955f6f4b6b3605f7ef00ee1_r.jpg" alt="img" style="zoom: 50%;" />

  - The second Multi-Head Attention
  
    **To be mentioned, Q is calculated from the last Decoder block.**
  
    **K, V is calculated from the Encoder matrix C.**
  
    
  
  - Softmax output Probabilities
  
    The output of the Decoder is the matrix Z, with n*d dimension.
  
    Each row represent a word's information, and the i th row only contains 1~i words' information.
  
    Use the i th row to predict i+1 word with softmax.



#### RNN & Transformer

**RNN**

The main features are:

1. Sequential processing: Sentences must be processed word for word

2. RNN refers to that the current output of a sequence is also related to the previous output. The specific expression is that the network will remember the previous information, save it in the internal state of the network, and apply it in the calculation of the current output, that is, the nodes between the hidden layers are no longer connected but connected. And the input of the hidden layer contains not only the output of the input layer but also the output of the hidden layer at the previous time

3. It adopts a linear sequence structure to continuously collect input information from front to back



**Transformer**

The main features are:

1. Non-sequential processing: Sentences are processed as a whole, not word-for-word

2. A single Transformer Block consists of two parts: Multi-Head Attention mechanism and Feed Forward neural network, Transformer Block replaces LSTM and CNN structure as our feature extractor. Make Transformer not rely on past hidden states to capture dependencies on previous words, but instead process a sentence as a whole to allow parallel computation, reduce training time, and reduce performance degradation due to long-term dependencies
