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


# # Konfidenzintervall

# 1. Berechnen Sie das $95 \%$ Konfidenzintervall umd den Mittelwert einer Normalverteilung $N(-2,3)$ 
# 2. Stellen Sie die Normalverteilung inklusive des Konfidenzintervalls dar.

# -------------------------------------------------------

# In[2]:


# Frage 1 ...


# In[3]:


# Frage 2 ...


# ### Lösung

# In[4]:


from scipy.stats import norm

loc = -2
scale = 4
alpha = 0.05

upper = norm.ppf(1 - alpha / 2, loc, scale)
lower = norm.ppf(alpha / 2, loc, scale)
print(f"Upper {int((1-alpha)*100)}%-CI: {upper}")
print(f"Lower {int((1-alpha)*100)}%-CI: {lower}")


# In[5]:


import matplotlib.pyplot as plt
import numpy as np

n = 100
xaxis = np.linspace(-20, 20, n)

fig, ax = plt.subplots()
ax.plot(
    xaxis,
    norm.pdf(xaxis, loc=loc, scale=scale),
    label=r"Wahrscheinlichkeitsdichtefunktion",
)
ax.axvline(upper, color="C1", linestyle="dashed", label="Upper CI")
ax.axvline(lower, color="C2", linestyle="dashed", label="Lower CI")
ax.legend()

