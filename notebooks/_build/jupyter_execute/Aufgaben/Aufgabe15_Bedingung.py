#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Load the "autoreload" extension
get_ipython().run_line_magic('load_ext', 'autoreload')
# always reload modules
get_ipython().run_line_magic('autoreload', '2')
# black formatter for jupyter notebooks
#%load_ext nb_black
# black formatter for jupyter lab
get_ipython().run_line_magic('load_ext', 'lab_black')

get_ipython().run_line_magic('run', '../src/notebook_env.py')


# # Aufgabe 2

# Welche Bedingung müssen die Zufallsvariablen $X_{1},X_{2}\dots,X_{n} $ erfüllen damit der zentrale Grenzwertsatz angewendet werden kann?

# ```{toggle}
# Die Zufallsvariablen müssen unabhängig und identisch verteilt sein. Erwartungswert $\mu$ und Standardabweichung $\sigma$ müssen existieren und endlich sein. Zwei Zufallsvariablen $X_1$ , $X_2$ die Elemente der Ereignisräume $X_1 \in B_1$ , $X_2 \in B_2$ sind, sind unabhängig wenn für die Wahrscheinlichkeiten gilt $P(X_1 \in B_1 , X_2 \in B_2) = P(X_1 \in B_1) \cdot P(X_2 \in B_2)$"
# ```

# In[ ]:




