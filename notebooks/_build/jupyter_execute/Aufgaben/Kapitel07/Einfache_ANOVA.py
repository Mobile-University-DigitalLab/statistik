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


# # Einfaktorielle ANOVA

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from numpy.random import normal

import pylab
from scipy.stats import t

from scipy.stats import uniform
from scipy import stats
from scipy.stats import f_oneway


# 1. Führen Sie jeweils eine schrittweise einfaktorielle Varianzanalyse für die folgenden Daten durch:
# * `sample_dat1`
# * `sample_dat2`
# * `sample_dat3`
# * `sample_dat4`
# * `sample_dat5`
# * `sample_dat6`

# \begin{array}{l}
# \hline
# \ \text{Schritt 1}  & \text{Geben Sie die Nullhypothese } H_0 \text{ und alternative Hypothese } H_A \text{ an.}\\
# \ \text{Schritt 2}  & \text{Legen Sie das Signifikanzniveau, } \alpha\text{ fest.} \\
# \ \text{Schritt 3}  & \text{Berechnen Sie den Wert der Teststatistik.} \\
# \ \text{Schritt 4} &\text{Bestimmen Sie den p-Wert.} \\
# \ \text{Schritt 5} & \text{Wenn }p\le \alpha \text{, } H_0 \text{ ablehnen } \text{; ansonsten } H_0 \text{ nicht ablehnen} \text{.} \\
# \ \text{Schritt 6} &\text{Interpretieren Sie das Ergebnis des Hypothesentests.} \\
# \hline 
# \end{array}

# 
# 
# - Benutzen Sie für Schritte $3$ und $4$ die Funktion `f_oneway()` die Sie mit `from scipy.stats import f_oneway` importieren können 
# 
# 2. Interpretieren Sie das Ergebnis

# In[3]:


from scipy.stats import norm, t

rs = 1
sample_dat1 = norm.rvs(loc=0, scale=1, size=25, random_state=rs)
sample_dat2 = norm.rvs(loc=0.01, scale=1.1, size=30, random_state=rs)
sample_dat3 = norm.rvs(loc=-0.01, scale=1.1, size=28, random_state=rs)
sample_dat4 = t.rvs(df=33, loc=0.8, scale=0.8, size=34, random_state=rs)
sample_dat5 = t.rvs(df=26, loc=0.5, scale=1.22, size=27, random_state=rs)
sample_dat6 = norm.rvs(loc=0, scale=1, size=25, random_state=rs)


# -------------------------------------------------------

# In[4]:


# Frage 1 ...


# ## Lösungen

# In[5]:


from scipy.stats import f_oneway

alpha = 0.01
statistics, pvalue = f_oneway(
    sample_dat1, sample_dat2, sample_dat3, sample_dat4, sample_dat5, sample_dat6
)

print(f"Wert der F-Statistik: {statistics}")
print(f"p-Wert: {pvalue}")

pvalue < alpha


# ```{toggle}
# Basierend auf den vorliegenden Daten zeigt die ANOVA einen signifikanten Unterschied zwischen den Datensätzen. Wir verwerfen also die Null-Hypothese.
# ```
