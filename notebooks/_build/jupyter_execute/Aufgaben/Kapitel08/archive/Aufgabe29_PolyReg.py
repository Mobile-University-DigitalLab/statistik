#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 8
# ### Aufgabenstellung 29 : Polynomiale Regression
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Führen Sie eine polynomiale Regression $2$-ten Grades für die folgenden Daten (`dat_x,dat_y`) durch:

# In[41]:


dat_x = np.array([0,  1,  2,  3,  4,  5,  6,   7,   8,   9,   10])
dat_y = np.array([1,  6,  17, 34, 57, 86, 121, 162, 209, 262, 321])


# und stellen Sie die Regressionsgerade und die Daten graphisch dar.

# -------------------------------------------------------

# ### Lösung

# In[42]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm


# In[43]:


X = dat_x.reshape(-1,1)
y = dat_y.reshape(-1,1)


# Polynomial Fit
poly = PolynomialFeatures(degree=2)
X_2 = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_2, y)

X_predict_reg_line = poly.transform(np.linspace(0,10, 25).reshape(-1,1))
y_reg_line = model.predict(X_predict_reg_line)

fig, ax = plt.subplots()
ax.plot(np.linspace(0,10, 25), y_reg_line, label='Regressionlinie')
ax.scatter(x=X, y=y,
           alpha=0.5,  color="white", edgecolor = 'k', label='Beobachtungen')

ax.legend();


# In[ ]:




