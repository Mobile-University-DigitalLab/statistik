#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 4
# ### Aufgabenstellung: Schätzfehler,Stichprobenfehler
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Berechnen Sie aus der gegebenen Grundgesamtheit $G$:

# In[1]:


pop = [4, 10, 10, 17, 5, 1, 11, 3, 15, 8, 10, 2, 11, 10, 15, 5, 0, 14, 1, 12]


# 1. den Mittelwert der Grundgesamtheit
# 2. den Mittelwert und Schätzfehler einer Stichprobe mit Umfang $n = 10$
# 3. die untere und obere Grenze für den Schätzfehler einer Stichprobe mit Umfang $n = 4$

# -------------------------------------------------------

# ### Lösung

# **1.**

# In[2]:


import numpy as np
pop_mean = np.mean(pop)
pop_mean


# **2.**

# In[3]:


import random
sample = random.sample(pop,10)
sample


# In[4]:


sample_mean = np.mean(sample)
print(sample_mean, 'Stichprobenmittel')
print(sample_mean-pop_mean, 'Schätzfehler')


# **3.**

# In[5]:


pop = sorted(pop)
pop


# In[6]:


# Entnehme kleinste mögliche Stichprobe n = 4
sample_min = pop[0:4]
sample_min


# In[7]:


print(np.mean(sample_min) - pop_mean,': Schätzfehler - untere Grenze')


# In[8]:


# Entnehme größte mögliche Stichprobe n = 4
sample_max = pop[16:20]
sample_max


# In[9]:


print(np.mean(sample_max) - pop_mean,': Schätzfehler - obere Grenze')


# In[ ]:




