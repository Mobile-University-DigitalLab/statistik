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


# # Binomialverteilung

# In einer Fabrik werden serienmäßig Tübel mit einem Ausschussanteil von $4,5\%$ hergestellt, d. h. unter $1000$ hergestellten Tübel befinden sich im Mittel genau $45$ unbrauchbare Tübel. 
# 
# 1. Mit welchen Wahrscheinlichkeiten finden wir in einer Zufallsstichprobe von $10$ Tübel genau $0, 1, 3, 5, 8$ unbrauchbare Tübel?
# 2. Mit welchen Wahrscheinlichkeiten finden wir in einer Zufallsstichprobe von $10$ Tübel $3$ oder mehr unbrauchbare Tübel?

# In[2]:


# Frage 1 ...


# In[3]:


# Frage 2 ...


# ## Lösungen

# 
# Es handelt sich hier um ein Bernoulli-Experiment. Die Wahrscheinlichkeit, beim Ziehen eines Tübels einen unbrauchbaren Tübels zu erhalten, liegt bei $p=0,04$. Die Zufallsvariable folgt der Binomialverteilung mit den Parametern $n=10$ und $p=0,4$.
# 
# Somit ergbit sich 
# 
# $$P(X=x)= \binom{10}{x} \times 0,04^x \times (1-0,04)^{(10-x)} \qquad (x=0,1,3,5,8)$$
# 

# In[4]:


from scipy.stats import binom

for x in [0, 1, 3, 5, 8]:
    rv = binom.pmf(k=x, n=10, p=0.04)
    print(f"For x={x} the probabilty is {rv}")


# $$P(X \geq3) = 1 - P(X =0) - P(X =1) - P(X =2)$$ 

# In[5]:


from scipy.stats import binom

# version 1
rv = 0
for x in [0, 1, 2]:
    rv += binom.pmf(k=x, n=10, p=0.04)
print(f"[version 1] The probabilty is is {1 - rv}")

# version 2
rv = 1 - binom.cdf(k=2, n=10, p=0.04)
print(f"[version 2] The probabilty is is {rv}")

