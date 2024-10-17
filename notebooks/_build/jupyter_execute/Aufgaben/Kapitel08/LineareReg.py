#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Load the "autoreload" extension
get_ipython().run_line_magic('load_ext', 'autoreload')
# always reload modules
get_ipython().run_line_magic('autoreload', '2')
# black formatter for jupyter notebooks
# %load_ext nb_black
# black formatter for jupyter lab
get_ipython().run_line_magic('load_ext', 'lab_black')

get_ipython().run_line_magic('run', '../../../src/notebook_env.py')


# # Lineare Regression

# 1. Führen Sie eine lineare Regression für die folgenden Daten (`x,y`) durch und stellen Sie die Regressionsgerade und die Daten graphisch dar.

# In[2]:


import numpy as np

n = 10
noise = np.random.normal(0, 1.4, n)
x = np.arange(0, n, 1)
y = 2 * x + noise


# -------------------------------------------------------

# In[3]:


# Frage 1 ...


# ## Lösungen

# In[4]:


import matplotlib.pyplot as plt
import statsmodels.api as sm

fig, ax = plt.subplots()
ax.scatter(x, y, label="Stichprobe")

# Fitte das Modell
model = sm.OLS(y, sm.add_constant(x)).fit()
x_axis = np.linspace(0, n, 100)
reg_line = model.predict(sm.add_constant(x_axis))
ax.plot(x_axis, reg_line, color="red", label="Regressionslinie")
ax.legend()
plt.show()

