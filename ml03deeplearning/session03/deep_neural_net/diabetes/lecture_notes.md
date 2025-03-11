
Lecture 03: Deep Neural Networks

1. DNN revision  
   1. DNN are non-linear function estimators and they use activation function to model the non-linearity in the data   
   2. NN learning / training using the forward/backward propagation   
   3. Forward propagation   
      1. Data passed into NN and wights are calculated and passed on to next layer  
      2. Finally a prediction is made   
   4. Loss function   
      1. Difference between actual and predicted value.   
      2. Log loss/ binary cross entropy function   
   5. Backward prop  
      1. Weights are adjusted to minimize the losses   
      2. So that predicted value is close to actual value.   
      3. Adjust the weights by implementing gradient descent   
      4. Global and local minima  
   6. Run the algo for multiple epochs so it reaches max accuracy  
   7. points to remember  
      1. Structure of NN  
      2. Operations within NN (summation and activation)  
      3. Forward propagation (data passing through NN and calculation at each neuron)  
      4. Loss function to estimate the loss  
      5. Backward propagation to adjust weights to reduce loss (gradient descent)  
      6. Multiple epochs of fwd/back propagation to achieve min loss.   
      7. Regularization to address overfitting problem   
      8. Keras was built on top of Tensorflow. now they are integrated   
2. DNN and Tools (17:50)  
   1. Explain DNN  
   2. Design a DNN (build and train the model)  
      1. Binary classification problem (diabetics data set)  
      2. Multiclass classification problem (amnesty data set)  
   3. Choose a loss function for a DNN  
   4. Describe and work with DL tools   
3. DNN  
   1. 1 or 2 hidden layer is called as shallow NN  
   2. More than 2 hidden layers is called DNN  
   3. Each layer is a feature engg layer   
      1. Input layer is a FE layer for raw input data   
   4. st hidden layer builds the feature engg of inputs which are linear combination of feature produced by the input layer  
      1. Input to each neuron is a feature and associated weight   
      2. The first hidden layer becomes per set of features.   
   5. 2nd hidden layer is also linear combination of values computed by 1st hidden layer  
      1. It become second FE layer on top of the first layer  
   6. Analogy of DNN with ML@ PCA  
      1. In ML, 100 features with PCA in the first step with raw data inputs.  
      2. Do another second PCA where input is from the output of the first PCA  
   7. DNN is a layer of features till it processes a final output.   
4. DNN learning  
   1. When an image is provided, it is broken down into different types of features  
      1. First layer: determine edges: horizontal, vertical, etc.   
      2. Face: eyes, eyebrows, lips   
5. None.
6. Tensor flow (30:00:00)  
   1. Tensor is a generic names for arrays. 1-D arrays (scalers), 2+D arrays (vectors)  
   2. Highly optimized data structure.   
   3. Flow is graph of operations   
   4. Highly optimized library performs complex operations very fast  
   5. Benefits of using graphs  
      1. Parallelism: easy to parallel operations  
      2. Distributed execution: partition program to run distributed manner.  
      3. Compilation: generate faster code   
      4. Portability: data flow graph in python, train model, and use it in C++.  
   6. Benefits of TensorFlow  
      1. Flexibility   
      2. Parallel execution   
      3. Multiple environments   
      4. Large user community   
7. Tensor flow lab (39:00:00)  
   1. NN are sensitive to scaling so it is mandatory to do scaling. (53:40:00)  
   2. [Click here to see the demo code of this lecuture](https://github.com/anilj1/aiml/blob/master/ml03deeplearning/session03/diabetes/Deep%20Learning%20with%20Keras%20-%20Diabetes%2014Jan2023.ipynb)
8. Types of loss function (2:47:00)  
   1. What is the loss function?   
      1. Difference between the actual and predicted value   
      2. Tells the accuracy of the prediction model   
   2. Types of losses   
      1. Regression loss (continuous variables)   
         1. Mean square error (MSE)  
            1. Average squared difference between actual and predicted values  
            2. Sum of squared errors / N (total values)   
            3. Taking the square root of MSE becomes RMSE.   
            4. Since error is squared, it penalizes the small difference in prediction compared to MAE.   
         2. Mean absolute error   
            1. Calculates the difference between actual and predicted value and take absolute value of it.   
         3. When to use MSE and MAE?   
            1. To control the outliers in the errors, better to use MAE.   
      2. Classification loss (discrete variableS)   
         1. Cross entropy loss  
            1. Binary cross entropy (when output is of two types)  
               1. Cross Entropy (C) \= \-y \* log(y)     when y \= 1  
               2. Cross Entropy (C) \= \- (1-y) \* log(1-y^)     when y \= 0  
            2. Categorical cross entropy (when output is of multiple types)  
               1. Cross entropy \= \- (y1 \* log(y1^) \+ y2 \* log(y2^ \+ … \+ yN \* log(yN^))  
         2. Hinge loss   
9. Classification cross entropy Lab (2:58:00)  
   1. Using the M-NIST data set, which is handwritten digits. 
