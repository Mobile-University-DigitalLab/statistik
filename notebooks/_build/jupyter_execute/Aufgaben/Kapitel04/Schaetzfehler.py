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


# # Stichprobenfehler

# Berechnen Sie aus der gegebenen Grundgesamtheit $pop$:

# In[2]:


pop = [4, 10, 10, 17, 5, 1, 11, 3, 15, 8, 10, 2, 11, 10, 15, 5, 0, 14, 1, 12]


# 1. den Mittelwert der Grundgesamtheit
# 2. den Mittelwert und Stichprobenfehler einer Stichprobe mit Umfang $n = 10$

# -------------------------------------------------------

# In[3]:


# Frage 1 ...


# In[4]:


# Frage 2 ...


# ## Lösungen

# In[5]:


import numpy as np

pop_mean = np.mean(pop)
pop_mean


# In[6]:


import random

random.seed(42)  # for reproducibility
sample = random.sample(pop, 10)
sample_mean = np.mean(sample)
print(f"Stichprobe:        {sample}")
print(f"Stichprobenmittel: {sample_mean}")
print(f"Schätzfehler:      {sample_mean-pop_mean}")

