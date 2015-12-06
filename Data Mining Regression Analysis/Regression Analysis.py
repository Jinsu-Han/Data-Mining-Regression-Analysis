
# coding: utf-8

#    #                Data Mining
#      
#    ###                     
#    ##                December 04, 2015

# # Regression Analysis
# Prediction is similar to classification but models continuous/Numerical/Orderd valued
# 
# #### About  
# In this notebook, we download a dataset that is related to fuel consumption and Carbon dioxide emission of cars. Then, we split our data into training and test sets, create a model using training set, Evaluate our model using test set, and finally use model to predict unknown value

# ### Importing Needed packages
# 
# Statsmodels is a Python module that allows users to explore data, estimate statistical models, and perform statistical tests.

# In[1]:

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().magic(u'matplotlib inline')


# #  Understanding the Data
# 
# ###  FuelConsumption.csv:
# 
# We have downloaded a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.
# 
# #### Data Source : [Data Source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64)

# * **MODELYEAR** e.g. 2014
# 
# * **MAKE** e.g. Acura
# 
# * **MODEL** e.g. ILX
# 
# * **VEHICLE CLASS** e.g. SUV
# 
# * **ENGINE SIZE** e.g. 4.7
# 
# * **CYLINDERS** e.g 6
# 
# * **TRANSMISSION** e.g. A6
# 
# * **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# 
# * **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# 
# * **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# 
# * **CO2 EMISSIONS (g/km)** e.g. 182 --> low --> 0

# ### Reading the data 

# In[4]:

df = pd.read_csv("E:/Data Collection/FuelConsumption.csv")

# take a look at the dataset
df.head()


# ###  Data Exploration

# In[5]:

# summarize the data
df.describe()


# In[9]:

# Check the shape of the data ( rows, columns)
df.shape


# In[10]:

# check the column names
df.columns


# In[13]:

# Create  new data frame CDF AND VIZ from existing dataframe
# New dataframe contains selected columns from old dataframe
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head()


# In[17]:

# creatre data frame VIZ with selected columns from existing cdf dataframe
viz= cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist


# In[18]:

plt.show()


# In[20]:

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()


# In[21]:

plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color ='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emissions")
plt.show()


# ###  Creating train and test dataset

# In[22]:

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# ##  Simple Regression Model
# 
# ##  What about linear regression?
# LinearRegression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' between the independed x in the dataset, and the dependend y by the linear approximation.

# ###  Train data distribution

# In[24]:

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color = 'blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()


# ### Modeling
# 
# **Using sklearn package to model data**

# In[25]:

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)

# The coefficients

print 'Coefficients : ',regr.coef_
print 'Intercept : ',regr.intercept_


# ### Plot Outputs

# In[26]:

train_y_ = regr.predict(train_x)
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS, color ='blue')
plt.plot(train_x, train_y_, color='black', linewidth =3)


# ###  Evaluation
# 
# **Evaluate the model with the Test data**

# In[28]:

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print ("Residual Sum of Squares : %.2f"
      % np.mean((test_y_ - test_y)**2))

# Explained variance score : 1 is perfect prediction

print('Variance Score : %.2f' % regr.score(test_x,test_y))



# ### Plot Outputs

# In[29]:

plt.scatter(test_x,test_y, color ='blue')
plt.plot(test_x, test_y_, color ='black', linewidth =3)
plt.xlabel("Engine Size")
plt.ylabel("Emissions")
plt.show()


# ## Non-linear regression

# In[30]:

# Import necessary packages
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(2),Ridge())
model.fit(train_x,train_y)
train_y_ = model.predict(train_x)
plt.scatter(test_x,test_y, color = 'blue')
plt.scatter(train_x,train_y_,linewidth = 2)


# ## Multiple Regression Model

# In[33]:

# Import necessary packages

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)

#  The Coefficients

print 'Coefficients : ',regr.coef_


# In[34]:

y_=regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual Sum of Squares : %.2f"
     %np.mean((y_ -y)**2))

#Explained Variance Score : 1 is perfect prediction

print('Variance Score : %.2f' % regr.score(x,y))


# In[ ]:




# In[ ]:



