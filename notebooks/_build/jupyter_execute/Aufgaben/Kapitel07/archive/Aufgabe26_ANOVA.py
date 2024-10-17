#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 7
# ### Aufgabenstellung 26 : Multiples Testen : Bonferroni Korrektur
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Führen Sie einen post-hoc Mehrfachhypothesentests bei den Daten aus Aufgabe $25$ (`sample_dat4,sample_dat5,sample_dat6`) durch um zu bestimmen welcher Datensatz sich unterscheidet.
# 
# 2. Berechnen Sie Bonferroni Korrektur für den Mehrfachhypothesentest von oben. Was ändert sich am Ergebnis?

# -------------------------------------------------------

# ### Lösung 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from numpy.random import normal

import pylab
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import uniform
from scipy import stats
from scipy.stats import f_oneway

sample_dat4 = t.rvs(df = 33 , loc = 0.8 , scale = 0.8, size = 34,random_state = 1)
sample_dat5 = t.rvs(df = 26 , loc = 0.5 , scale = 1.22, size = 27,random_state = 1)
sample_dat6 = norm.rvs(loc=0,scale=1,size = 25,random_state = 1)


# In[2]:


alpha = 0.05

print(stats.ttest_ind(sample_dat4,sample_dat5))
statistics,p_value=stats.ttest_ind(sample_dat4,sample_dat5)
print('**Reject:**', p_value <= alpha)
                         
print(stats.ttest_ind(sample_dat4,sample_dat6))
statistics,p_value=stats.ttest_ind(sample_dat4,sample_dat6)
print('**Reject:**', p_value <= alpha)
                         
print(stats.ttest_ind(sample_dat5,sample_dat6))
statistics,p_value=stats.ttest_ind(sample_dat5,sample_dat6)
print('**Reject:**', p_value <= alpha)
                         


# Der Mehrfachvergleich ergibt für `sample_dat4-sample_dat5` keine signifikanten Unterschiede bei $\alpha = 0.05$

# **2.**
# 
# Die Bonferroni Korrektur ergibt sich zu:
# 
# $$\alpha = \frac{\alpha}{m}\text{,}$$
# 
# $$m=\frac{k(k-1)}{2}\text{,}$$
# 
# $$\alpha = \frac{0,05}{3}$$

# In[3]:


alpha = 0.05/3

print(stats.ttest_ind(sample_dat4,sample_dat5))
statistics,p_value=stats.ttest_ind(sample_dat4,sample_dat5)
print('**Reject:**', p_value <= alpha)
                         
print(stats.ttest_ind(sample_dat4,sample_dat6))
statistics,p_value=stats.ttest_ind(sample_dat4,sample_dat6)
print('**Reject:**', p_value <= alpha)
                         
print(stats.ttest_ind(sample_dat5,sample_dat6))
statistics,p_value=stats.ttest_ind(sample_dat5,sample_dat6)
print('**Reject:**', p_value <= alpha)
                         


# Der Mehrfachvergleich ergibt für `sample_dat4-sample_dat5` und `sample_dat5-sample_dat6` keine signifikanten Unterschiede bei Bonferroni korrigiertem $\alpha_B = \frac{0,05}{3}$.

# In[ ]:




