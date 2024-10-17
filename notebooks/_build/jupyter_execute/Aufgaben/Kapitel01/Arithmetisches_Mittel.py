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


# # Arithmetisches Mittel

# diedieSchreiben Sie eine Funktion (`arithmic_average`) die den Mittelwert einer Liste/eines Arrays berechnet. Vergleichen Sie Ihre Implementierung mit einer _built-in_ Funktion von Python.
# Verwenden Sie die Funktion auf die folgenden Daten an.

# In[2]:


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [-10, -8, -6, -4, -2, 0]


# -------------------------------------------------------

# In[3]:


# Frage 1 ...


# ## Lösungen

# In[4]:


def arithmic_average(values):
    """Function to compute the arithmic average"""
    return sum(values) / len(values)


print("Self implementation")
print(arithmic_average(a))
print(arithmic_average(b))
print("\nnumpy built-in function")
import numpy as np

print(np.mean(a))
print(np.mean(b))

