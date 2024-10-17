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

n = 100
x = np.linspace(-10,6,n)
upper = -2+norm.ppf(0.975)*2
lower = -2+norm.ppf(0.025)*2
fig, ax = plt.subplots()
ax.axvline(x=lower, ymin=0, ymax=1, label = 'Untere Grenze',color = 'k',linestyle = '--')
ax.axvline(x=upper, ymin=0, ymax=1, label = 'Obere Grenze',color = 'k',linestyle = '--')

ax.plot(x,norm.pdf(x, loc=-2, scale=2), label = 'Normalverteilung')
plt.legend()


# In[ ]:




