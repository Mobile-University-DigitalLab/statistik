#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # Aufgabe 1

# Erklären Sie den Zentralen Grenzwertsatz.

# ```{toggle}
# Zentraler Grenzwertsatz (informell)
# 
# Die Summe von unabhängigen Zufallsvariablen besitzt eine Verteilung, die
# sich durch eine Normalverteilung approximieren lässt, sofern die Anzahl der
# Summanden gross ist. Die Approximation gelingt um so besser, je grösser
# die Anzahl der Summanden ist {cite:p}`weigand2006statistik`.
# ```
# 

# In[ ]:




