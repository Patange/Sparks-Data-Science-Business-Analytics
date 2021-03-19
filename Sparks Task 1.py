#!/usr/bin/env python
# coding: utf-8

# # Task 1: Prediction using Supervised ML.
# 
# # Name: Vedika Patange
# 
# # Organisation: The Sparks Foundation
# 

# In[32]:


# Importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()


# In[4]:


# Reading and loading data from remote link
data = pd.read_csv('http://bit.ly/w-data')


# In[5]:


data


# In[28]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[16]:


#Divide the data set into it's attributes and values
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[17]:


#Split the data into training and testing set
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# In[18]:


#Fit the training data into regressor and train the model
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[19]:


# Plot the regression line
line = regressor.coef_*X+regressor.intercept_

# Plot for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[20]:


# Making Predictions
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[21]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[22]:


#Visualization of the training set
plt.scatter(X_train, y_train, color = 'purple')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Scores vs Hours of study (train set)')
plt.xlabel('Hours of study')
plt.ylabel('Scores')
plt.show()


# In[23]:


#Visualization of the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Scores vs Hours of study(Test set)')
plt.xlabel('Hours of study')
plt.ylabel('Scores')
plt.show()


# In[25]:


# Testing for 9.25 hours of work
hours = 9.25
print("No of Hours = {}".format(hours))
regressor.predict([[9.25]])


# In[26]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[27]:


#Evaluating the model by testing for it's accuracy
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# # Thank You
