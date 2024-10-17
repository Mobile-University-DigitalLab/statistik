#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 3
# ### Aufgabenstellung 11 : Normalverteilung
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Welche Kenngrößen charakterisieren die Normalverteilung
# 2. Was ist die Standardnormalverteilung und in welchem Bezug steht sie zur Normalverteilung?
# 3. Generieren Sie $10.000$ Zufallswerte für die Normalverteilung mit Mittelwert $\mu = 1$ und Standardabweichung $\sigma = 3$, unter Verwendung der Funktion `np.random.normal(loc, scale, size)`, und stellen Sie das Ergebnis als Histogramm dar.
# 4. Führen Sie eine $z$-Transformation für diese Werte durch und plotten Sie das Ergebnis.

# -------------------------------------------------------

# ### Lösung

# **1.**
# 
# Die Normalverteilung wird durch ihren Mittelwert $\mu$ und ihre Standardabweichung $\sigma$ beschrieben. $X \sim N( \mu, \sigma)$

# **2.** 
# 
# Die Standardnormalverteilung ist die Normalverteilung mit Mittelwert $\mu=0$ und Standardabweichung $\sigma=1$. Sie entspricht also der Normalverteilung mit $X \sim N( 0, 1)$. Durch die Anwendung der $z$-Transformation $z = \frac{x-\mu}{\sigma}$ können beliebige Normalverteilungen auf die Standardnormalverteilung abgebildet werden.

# **3.**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(loc=2, scale=2, size=10000)
x = np.linspace(-10,10,10000)
mean_normal = np.mean(data)
print('Mittelwert',mean_normal)

std_normal = np.std(data, ddof=1)
# Standardabweichung der Probe finden
print('Standardabweichung',std_normal)
plt.hist(data, 100);


# **4.**

# In[2]:


z = (data - mean_normal)/std_normal
plt.hist(z, 100);


# In[ ]:




