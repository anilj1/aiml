#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:
df_diab = pd.read_csv('../datasets/diabetes.csv')


# In[3]:
df_diab.head()


# In[4]:
df_diab.shape


# In[5]:
df_diab.Outcome.unique()


# In[6]:
# Check if any zero or missing values.
df_diab.isna().sum()


# In[7]:
# Baking the data set to extract dependent and independent variables.
X = df_diab.drop(['Outcome'],axis=1)
y = df_diab['Outcome']


# In[8]:
# A seed is a number that initializes the selection of numbers by a random number generator;
# given the same seed number, a random number generator will generate the same series of
# random numbers each time a simulation is run.
np.random.seed(12345)


# In[9]:
from sklearn.model_selection import train_test_split


# In[10]:
# Split into training and testing, to perform stratified sammpling
# Stratified sampling: dividing the whole data set into homogeneous groups called strata (stratum).

# train_size is 80% // 80% of the data is used to train the model. Large data is required for the training.
# random_state is 12345 //
# stratify = y // to get a balanced data for the training. Can only be done on the dependent data (not on independent data).
# the stratify sampling will ensure that the percentage of the split data is also same as the original data.
# Random sampling may create certain imbalances in the training and test data and may lead to inaccuracies.
X_train, X_test, y_train,y_test = train_test_split(X,
                                                   y,
                                                   stratify = y,
                                                   test_size=0.2,
                                                   random_state=12345)


# In[11]:
X_train.shape


# In[12]:
X_test.shape


# In[13]:
# Scale of certain data columns are different than each other. To unify the scale, standard scaling is performed.
# Other scaler types are: Min-Max scaler, Robust scaler.
from sklearn.preprocessing import StandardScaler


# In[14]:
# X-scaled = (X - mean)/sd -> z value

# Neural network are sensitive to scale so scaling is mandtory.
# Some ML algorithms also needs scaling

# (x-mu)/sigma
# Calculate Z values for each column divided by standard deviation.

# Training dataset is used for training the model. So 'mu' and 'sigma' is inferred from the training set only.
# Test dataset is used for validating the model. We keep the test set prstine. Don't learn or infer anything.

# Do fit_transform only on X_train and transform on X_test.
# We take the mean and standard deviation from X_train and apply it to X_test.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[15]:
import tensorflow as tf


# In[16]:
print(tf.__version__)


# In[17]:
# In TF, models are build using 'Seqiential' APIs and 'Functional' APIs.
# In 'sequential' APIs, we are building one layer (input, hidden-1, hidden-2, etc.) after the other.
from tensorflow.keras.models import Sequential


# In[18]:
# Every layer can be modeled as a 'Dense' layer.
# The 'Dense' layer a.k.a. fully connected layer i.e. every neuron is connected to previous layer and following layer.
# The 'Input' Layer
from tensorflow.keras.layers import Dense, Input


# In[19]:
# Instantiate a blank model. No layers are added as yet.
model = Sequential()


# In[20]:
# We are having 8 input features.
X_train_scaled.shape[1]


# In[21]:
# Add the all input features to the input layer.
# Layer has a 'shape' argument, where we initialie it with a 2-D array.
model.add(Input(shape = (X_train_scaled.shape[1],))) # Input Layer


# In[22]:
# First Hidden Layer - with 128 neurons, and activation function 'Relu'
# The number of first hidden layers can be anything. They will keep reducing more layers are added.
model.add(Dense(units=128,
                activation='relu'
         ))


# In[23]:
# Add two more hidden layer to make it a Deep neural network (DNN)
# Second Hidden Layer - 64 neurons, Relu activation
model.add(Dense(units=64, activation='relu'))

# Third Hidden Layer 32, relu activation
model.add(Dense(units=32, activation='relu'))


# In[24]:
# Output layer - Sigmoid.
# WE have addd only one neuron and using 'sigmoid' activation function
# in the output layer as it is a binary classification.
model.add(Dense(units=1, activation='sigmoid'))


# In[25]:
model.summary()


# In[26]:
#(8+1)*128 = 1152
#(128+1)*64 = 8256
#(64+1)*32 = 3080
#(1+1)*1 = 2


# In[27]:
# Note that in the first hidden (dense) layer, there are 128 neurons i.e. 128 * 8+1 input neurons.
# Therfore 128 * 9 = 1152 parameters (or links)

# Note that in the second hidden (dense) layer, there are 64 neurons i.e. 64 * 128+1 input neurons.
# Therfore 64 * 129 = 8256 parameters (or links)

# Note that in the third hidden (dense) layer, there are 32 neurons i.e. 32 * 64+1 input neurons.
# Therfore 32 * 65 = 2080 parameters (or links)

# Note that in the final output (dense) layer, there is 1 neurons i.e. 1 * 1+1 input neurons.
# Therfore 1 * 2 = 2 parameters (or links)

# Trainable parameters: the one which are able to optimize using the gradient descent algorithm.
1152+8256+2080+33+2


# In[28]:
# Compile the configured model.
# "adam" optimizer is used, which is the advanced version of gradeint descent.
# Use the binary cross entropy log loss function.
# Metric to be monitored for is 'accuracy'.
model.compile(optimizer='adam', # Variant of Gradient Descent
              loss= 'binary_crossentropy', # Log Loss or Binary Cross Entropy
              metrics=['accuracy']) # Monitor Accuracy


# In[29]:
# The function 'fit' trains the model where the model learns patterns from the provided data.
# X and Y: the input data and the corresponding target labels.
# validation_data: used to monitor model's performance during training, helping to detect overfitting.
# epochs: the number of times the model will iterate over the entire training dataset.
#  - Nunber of epochs is not fixed. Sometimes accuracy is achieved with a smaller value as well.
# Performs:
# - Feeding data to the model.
# - Calculating the error (loss).
# - Adjusting the model's parameters to reduce the error.
# - Repeating this process for a specified number of times.
result = model.fit(X_train_scaled,
                   y_train,
                   validation_data = (X_test_scaled, y_test),
                   epochs=100)

# Interpreting output:
# Epoch 88/100
# 20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.8610 - loss: 0.4383 - val_accuracy: 0.7597 - val_loss: 0.5397
# - accuracy: on the training data set.
# - loss: loss on the training data set.
# - val_accuracy: on the validation data set.
# - val_loss: loss on the validation data set.

# The large difference beween the training accuracy and the validation accuracy is the indicator of overfitting.
# Accuracy depends on number of layers.
# With CNN, we can not train for a larger number of epochs (e.g. 50+). We need a higher compute (GPU/TPU) machine.
# Auto encoders are trained using 500+ epochs e.g. complex data (e.g. music).


# In[30]:
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)


# In[31]:
y_train_pred[0] >=0.5


# In[32]:
# Import the confusion matrix and ccuracy score to determine the loss (correlating it to the model)
from sklearn.metrics import confusion_matrix, accuracy_score


# In[33]:
confusion_matrix(y_pred=y_train_pred >= 0.5,
                 y_true = y_train)


# In[34]:
# Verify the accuracy on the training data set.
accuracy_score(y_pred=y_train_pred >= 0.5,
               y_true = y_train)


# In[35]:
confusion_matrix(y_pred=y_test_pred >= 0.5,
                 y_true = y_test)


# In[36]:
# Verify the accuracy on the test data set.
accuracy_score(y_pred=y_test_pred >= 0.5,
               y_true = y_test)


# In[37]:
# The record of the training process, specifically the metrics that were tracked during each epoch.
# These are the same numbers which are displayed during the training of the model at each epoch.
history = pd.DataFrame(result.history)


# In[38]:
history.head()


# In[39]:
import matplotlib.pyplot as plt


# In[40]:
# Plot the graph of the history to visualize the difference between the training loss and the testing loss.
plt.figure(figsize = (25,6))
plt.plot(history.val_loss, label='Test Set Loss')
plt.plot(history.loss, label='Training Set Loss')
plt.title('Epochs vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[41]:
# Plot the graph of the history to visualize the difference between the training accuracy and the testing accuracy.
plt.figure(figsize = (15,6))
plt.plot(history.val_accuracy, label='Test Set Accuracy')
plt.plot(history.accuracy, label='Training Set Accuracy')
plt.title('Epochs vs Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[42]:
# Display the model weights layer by layer.
model.get_weights()


# In[43]:
# Model accuracy depends on:
# - the number of layers in the model.
# - the neurons in each of the layer.
# - Process is called experimentation.
# - With experience, the number of layers and neurons in each layer
# - Hyperparameter tuning using random search to find optimal neurons and activation functions.
#   - keras.tuner library for this purpose. Use this as only as a reference.
# - Grid search
# With optimal values we are trying to impact the accuracy of the model.
# With regulerization, we are making sure that the overfitting does not happen.
# - Regulirization adds an additional term to the loss function (sum of square of the weights)
# - It puts constrains on the parameters at each layer by adding the sum of square of the weights.
# - This ensures that the weights do not go very high.


# # Implement Early Stopping

# In[45]:
# Early stopping is the point at which there is no improvement in the accuracy of the model. The model starts overfitting.
# A call back function check for this condition and stops the model training.
# Learned hypothesis may fit the training data and the outliers (noise) very well but fail to generalize the test data.
# A check to see if the validation loss is not improving, do not train any further and stop the training.
from tensorflow.keras.callbacks import EarlyStopping


# In[46]:
model_es = Sequential()


# In[47]:
model_es.add(Input(shape = (X_train_scaled.shape[1],)))


# In[48]:
model_es.add(Dense(units = 128, activation='relu'))


# In[49]:
model_es.add(Dense(units = 64, activation = 'relu'))


# In[50]:
model_es.add(Dense(units = 32, activation = 'relu'))


# In[51]:
model_es.add(Dense(units = 1, activation = 'sigmoid'))


# In[52]:
model_es.summary()


# In[53]:
model_es.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['Accuracy'])


# In[54]:
# Callbacks are used to implement early stopping programatically by injecting a custom code into tensor flow execution.
# Assuming 'es' is your EarlyStopping callback.

# Early stopping criteria:
# - Monitor the training accuracy of the model.
# - Check for the number of epochs given in the petience. Stop the model if desired results are not improving further.
es = EarlyStopping(
    monitor='val_Accuracy',
    mode='max',
    patience=5,
    restore_best_weights=True
) # adjust patience as needed.


# In[55]:
# Make sure that the model fitting is done on a blank model. If you run the fit model multiple time,
# it is going to use the weights from the last step and start again from there, which is not correct.
# On a blank model, it uses the weights initialized using the xavier's distribution.
# Logically, running fit method multiple times if like running those many epochs.
result_es = model_es.fit(X_train_scaled,
                         y_train,
                         validation_data = (X_test_scaled, y_test),
                         epochs = 100,
                         callbacks=[es]
                        )

# Using ES is a double edge sword.
# - If you get less number of patience, it may not give correct results.
# - For complex model, the learning starts quite late. So chosing a correct value is very important.


# In[56]:
history_es = pd.DataFrame(result_es.history)


# In[57]:
plt.figure(figsize = (15,6))
plt.plot(history_es.val_loss, label='Test Set Loss')
plt.plot(history_es.loss, label='Training Set Loss')
plt.title('Epochs vs Loss (Early Stopping)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# # Implementing L1 Regularization

# In[59]:
model_reg = Sequential()


# In[60]:
model_reg.add(Input(shape = (X_train_scaled.shape[1],)))


# In[61]:
# First Hidden Layer - 128 neurons, Relu activation, L1 Regularizer for the Kernel.
# The reguerization happens at the kernel i.e. at the first hidden layer level.
# No regulrization is applied at the subsequent laeyrs.

# L1 is callled Lasso regulerization
# L2 is called Ridge regulerization.
# L1 and L2 combined is called ElasticNet regulerization.
# Typically, L1 with early stopping or dropout is recommended approach.
model_reg.add(Dense(units=128,
                    activation='relu',
                    # kernel_regularizer = tf.keras.regularizers.L2()
                    kernel_regularizer = tf.keras.regularizers.L1()
                   )
             )


# In[62]:
# Second Hidden Layer - 64 neurons, Relu activation
# No regulrization is applied at the subsequent laeyrs.
model_reg.add(Dense(units=64, activation='relu'))


# In[63]:
# Third Hidden Layer 32, relu
model_reg.add(Dense(units=32, activation='relu'))


# In[64]:
# Output layer - Sigmoid
model_reg.add(Dense(units=1, activation='sigmoid'))


# In[65]:
model_reg.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])


# In[66]:
model_reg.summary()


# In[67]:
result_reg = model_reg.fit(X_train_scaled,
                           y_train,
                           validation_data = (X_test_scaled, y_test),
                           epochs=100)


# In[68]:
history_reg = pd.DataFrame(result_reg.history)


# In[69]:
plt.figure(figsize = (25,6))
plt.plot(history_reg.val_loss, label='Test Set Loss')
plt.plot(history_reg.loss, label='Training Set Loss')
plt.title('Epochs vs Loss (L1 Regularization)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[70]:
# Summary
# - When a model is deployed in production, we have to monitor for the accuracy of the model.
# - We check for if the distribution of the data changing.
# - For example, we are sending "Propensity to buy a product" in a marketing campaign refers to:
#   - the predicted likelihood that a specific customer will purchase a particular product,
#     based on their past behavior, demographics, and other relevant data.
#     buying power of customer (e.g. having more money), the model data is changing.
# - It is recommended that we update model eveyr 3-6 months for changing data.
#   - the model wights needs to be updated and model needs to be retrained to improve the accuracy.

# The real test of a mode is how it performs on an unseen data.
# It is not import that the loss lines intersect each other on the plot as long as their trend remain constant.


# # Implement Dropout Regularization

# In[100]
from tensorflow.keras.layers import Dropout


# In[102]:
model_drp = Sequential()


# In[104]:
model_drp.add(Input(shape = (X_train_scaled.shape[1],)))


# In[106]:
model_drp.add(Dense(units=128, activation='relu'))


# In[108]:
model_drp.add(Dropout(0.5))


# In[110]:
model_drp.add(Dense(units=64, activation='relu'))


# In[112]:
model_drp.add(Dropout(0.3))


# In[114]:
model_drp.add(Dense(units=32, activation='relu'))


# In[116]:
model_drp.add(Dropout(0.25))


# In[118]:
model_drp.add(Dense(units=1, activation='sigmoid'))


# In[120]:
model_drp.summary()

# We have added a dropout layer for each layer (except the output layer)
# Dropout layer has no parameters as they are not learning anything in there.
# We can not have dtopout layer at the input layer as it would drop the input features itself.


# In[122]:
model_drp.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])


# In[124]:
result_drp = model_drp.fit(X_train_scaled,
                           y_train,
                           validation_data = (X_test_scaled, y_test),
                           epochs=100)


# In[128]:
history_drp = pd.DataFrame(result_drp.history)


# In[130]:
plt.figure(figsize = (15,6))
plt.plot(history_drp.val_loss, label='Test Set Loss')
plt.plot(history_drp.loss, label='Training Set Loss')
plt.title('Epochs vs Loss (Dropout Regularization)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[132]:
# Summary
# - In real world, overfitting is the biggest problem we have to deal with.
# - May be the data distribution between training and test set is not good.
# - The way to tackle this is through experimentation.
# - Whether to combine L1, L2, early stopping, dropout, etc all depenends on experimentation.
