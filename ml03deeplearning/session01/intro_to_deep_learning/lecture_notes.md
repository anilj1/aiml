Deep Learning Introduction

Background on AI and ML

1. AI workflow   
   1. Learn   
   2. Reason  
   3. Self correct  
2. Supervised learning  
   1. Type of models and algorithms  
      1. Linear regression   
      2. Logistic regression   
      3. Decision tree  
      4. Random forest  
      5. Ensemble   
      6. Naive bayes  
      7. Support vector machines   
      8. Linear discriminant analysis   
   2. Basics of supervised learning  
      1. Labeled data i.e. historical data  
      2. Dependent and independent variables  
      3. Finds relation between the two variables  
      4. Mapping function, y \= f(x)  
      5. Predicting next value  
      6. Function should achieve certain level of performance to be fit for production   
      7. Binary classification   
      8. Multi class classification  
      9. Precision, recall, and accuracy for classification problem (discrete variables)  
      10. Mean squared error, mean absolute error, RMSC for regression problems (continuous variables)  
   3. Supervised learning algorithm details  
      1. Decision tree  
         1. Entropy   
      2. Classification problem (discrete values)  
      3. Regression problem  (continuous values)   
         1. Loss and loss functions   
         2. Y \= m1 \* x1 \+ m2 \* x2   
         3. Ordinary least square   
         4. Cross entropy  
         5. Fit a line and minimize the distance between the points and line   
      4. Logistic regression   
         1. Once the linear regression equation is found, it converts the linear values into probability   
         2. Sigmoid function (probability function) used to calculate output value of logistic regression   
         3. Converts the continuous value into a probabilistic value (0-1).  
         4. Probability of specific class based on the threshold to decide if class is 0 or 1 / yes or no.  
3. Unsupervised learning   
   1. Type of models and algorithms  
      1. Clustering   
         1. Hierarchical   
         2. K-means   
         3. Unique clusters based on centroids   
         4. Density-based spatial clustering of apps with noise (DBSCAN)  
      2. Apriori algorithm  
         1. Association rule mining  
         2. Recommendation engine   
         3. Market basket analysis   
      3. Principal component analysis (PCA)  
         1. Dimensionality reduction   
         2. Orthogonal principal components   
      4. Non-negative matrix factorization   
         1. Similar to what recommendation engine do  
         2. Topic modeling  
   2. Basics of supervised learning  
      1. There is only X-data, no Y-data,   
      2. Data driven i.e. find the patterns in the data  
         1. Data separation by forming of groups  
4. Reinforcement learning   
   1. Type of models and algorithms  
      1. Positive reinforcement   
      2. Negative reinforcement   
      3. Monte-carlo   
      4. Q-learning  
      5. State-action-reward-state-action (SARSA)  
      6. Deep Q-learning  
      7. Q-lambda   
      8. SARSA-Lambda  
   2. Basics of supervised learning  
      1. Formed on the basis of psychology   
         1. [Pavlov’s dog experiment](https://www.youtube.com/watch?v=AC8TaWRwc6s)   
      2. Learn from the trial and errors (mistakes)  
      3. Environment that provide reward and penalties   
      4. Moving the agent towards its goal (goal oriented)  
      5. Agent with a goal takes an action in an environment which gives the reward/penalty and changes the state of the agent.  
5. Deep learning   
   1. Type of models and algorithms  
      1. Artificial neural networks (ANN)  
         1. Utilizes ANN inspired by the biological neural network of the human brain.   
         2. Imitate neural activity of the brain to learn patterns.  
   2. Basics of deep learning  
      1. Machine learning: feature extraction is performed manually.  
      2. Deep learning: the feature extraction and classification is bundled inside the neural network  
      3. It is very difficult to explain why a feature was selected or dropped.   
         1. Some legal compliance requires to explain why/how the decision is taken.   
         2. Neural networks create and identify their own features.   
      4. Contrary, in supervised learning feature engineering is performed by a human.   
      5. Applications of deep learning   
         1. Used where the data set is very complex (e.g., image data).   
         2. Image classification, language translation are some of the classical deep learning problems  
         3. Generating a picture from a text prompt   
         4. Finding fraudulent transactions from thousands of transacitons.   
   3. Deep learning neuron model   
      ![png][https://github.com/anilj1/aiml/blob/master/ml03deeplearning/session01/intro_to_deep_learning/image1.PNG]  
      1. Input layer with features as input data   
         1. Each circle on the input layer denotes a specific type of data / feature. The data could be of any type.   
      2. Hidden layers with multiple neurons in each layer  
      3. Output layer with output prediction of decision   
         1. Output could be one or more neurons indicating a number of outputs.  
      4. Neuron is a computational unit. Each neuron is connected to every neuron in the previous layer and every neuron in the next layer.   
      5. Neural network types   
         1. Shallow neural network (1-2 hidden layers)  
         2. Deep neural network ( 2+ hidden layers)   
   4. Data scientist   
      1. To decide how many layers are required to solve a given problem.   
      2. Number of neurons in a layer is a function of the number of features.   
      3. Number of layers and \# neurons in a layer are hyperparameters of NN.  
   5. NLP  
      1. Auto completion   
      2. Associative reasoning   
      3. BARD, GPT, RoBERta   
   6. Performance of a NN model depends on the data size   
      1. Performance (time complexity and accuracy of the output)

      ![png][https://github.com/anilj1/aiml/blob/master/ml03deeplearning/session01/intro_to_deep_learning/image2.PNG]

      2. Number of records   
      3. Complexity of the data (tabular vs audio/video data)  
         1. ImageNet (14M labeled images)   
         2. AlexNet  
         3. Google Brain (70% improvement)   
         4. DeepFace (using 4M images with 97.35% accuracy)  
            1. Breakdown a big picture into smaller chunk and extract features from each small chunk of data   
6. Self driving cars  
   1. image classification and CNN  
   2. object localization, bounding box i.e., where the car is located Ina picture.  
7. Deep learning challenges at 49:00  
8. Computer vision 51:40  
   1. GAN generative adversarial network  
   2. Challenges 1:06:20  
   3. LSTM 1:15:15  
9. Lifecycle of DL project 1:17:00  
   1. Project planning 1:21:00
