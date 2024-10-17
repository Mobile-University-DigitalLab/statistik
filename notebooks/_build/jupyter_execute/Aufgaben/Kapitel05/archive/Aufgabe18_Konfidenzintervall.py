#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 5
# ### Aufgabenstellung 18 : Konfidenzintervall
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Berechnen Sie das $95 \%$ Konfidenzintervall für die Normalverteilung $N(-2,2)$ und stellen Sie die Normalverteilung inklusive des Konfidenzintervalls dar.

# -------------------------------------------------------

# ### Lösung

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

n = 1000
x = np.linspace(-10,10,n)
upper = norm.pdf(x, loc=0, scale=2) + norm.ppf(0.975,0,2)*np.std(norm.pdf(x, loc=0, scale=2))/np.sqrt(n)
lower = norm.pdf(x, loc=0, scale=2) + norm.ppf(0.025,0,2)*np.std(norm.pdf(x, loc=0, scale=2))/np.sqrt(n)
plt.plot(x,lower, label = 'Untere Grenze')
plt.plot(x,upper, label = 'Obere Grenze')
plt.plot(x,norm.pdf(x, loc=0, scale=2), label = 'Normalverteilung')
plt.legend()


# In[ ]:




