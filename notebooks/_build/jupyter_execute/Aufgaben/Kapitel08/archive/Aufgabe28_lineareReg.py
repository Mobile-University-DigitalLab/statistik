#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 8
# ### Aufgabenstellung 28 : Lineare Regression
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Führen Sie eine lineare Regression für die folgenden Daten (`x,y`) durch:

# In[1]:


import numpy as np
noise = np.random.normal(0,1.4,10)
x = np.arange(0,10,1) 
y = 2*x + noise


# und stellen Sie die Regressionsgerade und die Daten graphisch dar.

# -------------------------------------------------------

# ### Lösung

# In[2]:


import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[3]:


noise = np.random.normal(0,1.4,10)
x = np.arange(0,10,1) 
y = 2*x + noise


# In[4]:


plt.scatter(x,y)


# In[5]:


# Fitte das Modell
x2 = sm.add_constant(x)
model = sm.OLS(y, x2).fit()
# Definiere x-Achse
x_axis = sm.add_constant(np.linspace(0,10,50))
# Berechne Regressionsgerade
y_reg = model.predict(x_axis)
plt.scatter(x,y)
plt.plot(np.linspace(0,10,50),y_reg,color = 'red')


# In[ ]:





# In[ ]:





# In[ ]:




