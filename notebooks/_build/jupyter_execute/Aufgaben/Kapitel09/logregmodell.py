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


# # Einfaches logistisches Regressionsmodell 

# 1. Erstellen Sie ein einfaches logistische Regressionsmodell für die folgenden Daten in Python und stellen Sie das logistische Modell graphisch dar.

# In[2]:


x = [
    29,
    15,
    33,
    28,
    39,
    44,
    31,
    19,
    9,
    24,
    32,
    31,
    37,
    35,
    8,
    4,
    11,
    12,
    33,
    45,
    20,
    25,
    27,
    26,
    29,
]
y = [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]


# -------------------------------------------------------

# ## Lösungen

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


log_model = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial())
log_results = log_model.fit()

x_axis = np.linspace(min(x), max(x), num=100)
predictions = log_results.get_prediction(exog=sm.add_constant(x_axis)).summary_frame()


fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x_axis, predictions["mean"])

ax.grid()
ax.set_ylabel("Wahrscheinlichkeit")
plt.show()

