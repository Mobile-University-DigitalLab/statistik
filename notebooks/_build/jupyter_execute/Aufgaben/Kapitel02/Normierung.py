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


# # Normierung

# Eine Zufallsvariable $X$ kann die Werte $1,2,3,4$ annehmen. Angenommen $f(x) = P(X = x)$ ist die dazugehörige Wahrscheinlichkeitsfunktion und ist gegeben durch:
# 
# $$
#      f(x)=\left\{\begin{array}{ll} \frac{x (b - x)}{10}, & x\in \{ 1,2,3,4 \} \\
#          0, & \text{sonst}\end{array}\right. .
#   $$
#   
# Wie muß $b$ gewählt werden um die Normierung der Wahrscheinlichkeitsfunktion zu gewährleisten ?

# In[2]:


# Frage 1 ...


# ## Lösungen

# Um die Wahrscheinlichkeitsfunktion zu normieren muß 
# 
# $$\sum_{i=1}^n f(x_i) = 1$$ 
# gelten.
# 
# $$\sum_{i=1}^4 f(x_i) = \frac{1 \cdot (b-1)}{10} + \frac{2 \cdot (b-2)}{10} + \frac{3 \cdot (b-3)}{10} + \frac{4 \cdot (b-4)}{10} = 1$$
# $$ (b-1)+ (2b-4) + (3b-9) + (4b - 16)  = 10$$
# $$ b + 2b + 3b + 4b-1-4-9-16  = 10$$
# $$  10b-30  = 10 $$
# $$\Rightarrow b = 4$$
