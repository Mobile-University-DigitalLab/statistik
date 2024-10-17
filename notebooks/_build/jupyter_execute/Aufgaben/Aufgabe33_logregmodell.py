#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 9
# ### Aufgabenstellung 33 : Einfaches logistisches Regressionsmodell
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Erstellen Sie ein einfaches logistische Regressionsmodell für die folgenden Daten in Python.
# 
# `x = [29,15,33,28,39,44,31,19,9,24,32,31,37,35]` und
# `y = [0,0,1,1,1,1,1,0,1,0,0,0,1,1]`
# und stellen Sie das logistische Modell graphisch dar.

# -------------------------------------------------------

# ### Lösung

# In[1]:


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import glm


# In[2]:


x = [29,15,33,28,39,44,31,19,9,24,32,31,37,35,8,4,11,12,33,45,20,25,27,26,29]
y = [0,0,1,1,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,1,0,1,1,1,1]


# In[3]:


# Erstelle x-Achse für Vorhersagen
x_axis = np.linspace(min(x),max(x), num = 100)

x2 = sm.add_constant(x)
x_grid = x_axis
x_axis = sm.add_constant(x_axis)

log_model = sm.GLM(y, x2, family=sm.families.Binomial())
log_results = log_model.fit()
# Berechne Vorhersagen für x_axis
predictions = log_results.get_prediction(exog = x_axis).summary_frame()

# Formatiere Plots
fig, ax = plt.subplots()
ax.scatter(x,y)
ax.plot(x_grid,predictions['mean'])

#plt.grid()
plt.xlabel("x")
plt.ylabel("Wahrscheinlichkeit")
plt.show()


# In[ ]:




