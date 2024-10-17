#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 5
# ### Aufgabenstellung 19 : Punktschätzungen bei unbekanntem $\sigma$
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Welche Wahrscheinlichkeitsverteilung wird für die Berechnungen von Punktschätzungen bei unbekannter Standardabweichnung der Grundgesamtheit und kleiner Stichprobengrösse ($n\lt 30$) verwendet?
# 2. Wir simulieren eine Grundgesamtheit an Daten indem wir $100$ Zufallszahlen zwischen $1$ und $100$ generieren und diese in einer Liste speichern. Berechnen Sie Mittelwert und Standardabweichung für diese Zufallsdaten. 
# 3. Nehmen Sie eine Stichprobe vom Umfang $n=10$. Berechnen Sie Mittelwert und Konfidenzintervall ($\alpha = 0,05$) für die Stichprobe unter der Verwendung einer geeigneten Wahrscheinlichkeitsverteilung und überprüfen Sie ob der Mittelwert der Grundgesamtheit innerhalb des Konfidenzintervalls liegt.

# -------------------------------------------------------

# ### Lösung

# **1.**
# 
# Im Fall unbekannter Standardabweichung der Grundgesamtheit $\sigma$ kann die Standardabweichung der Stichprobe, $s$, als Schätzer verwendet werden.
# $s$ ist gegeben durch $s =\sqrt{\sum_i \frac{x_i - \bar x}{n-1}}$ und ihre Verteilung wird durch die $t$-Verteilung beschrieben.
# 
# ({cite:p}`fahrmeirstatistik` s.360)

# **2.**

# In[1]:


import numpy as np
import random
from random import seed

# Erzeuge Random seed
seed(1)
data = []
# Generiere Zufallszahlen
for i in range(0,100):
    n = random.randint(1,100)
    data.append(n)

# Berechne Mittelwert und Standardabweichung
data_mean = np.mean(data)
data_std = np.std(data)
print(data_mean, 'data_mean')
print(data_std, 'data_std')


# **3.**

# In[2]:


# Nehme Stichprobe
n = 10
sample_dat = random.sample(data,n)

# Berechne Mittelwert und Standardabweichung
sample_mean = np.mean(sample_dat)
sample_std = np.std(sample_dat)
print(sample_mean, 'sample_mean')
print(sample_std, 'sample_std')


# In[3]:


from scipy.stats import t
lower = data_mean - t.ppf(1-0.05/2,df = n-1)*(sample_std/np.sqrt(n))
lower


# In[4]:


upper = data_mean + t.ppf(1-0.05/2,df = n-1)*(sample_std/np.sqrt(n))
upper


# In[5]:


print(lower,data_mean,upper)
lower <= data_mean and data_mean <= upper


# In[ ]:




