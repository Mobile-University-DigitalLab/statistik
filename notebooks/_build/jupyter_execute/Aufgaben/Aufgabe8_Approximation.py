#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 2
# ### Aufgabenstellung 8 : Binomiale Approximation der Poissonverteilung
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Wie lauten der Mittelwert und Standardabweichung für Binomial- und Poissonverteilung?
# 2. Unter welchen Voraussetzungen approximiert die Binomialverteilung die Poissonverteilung $P(X = x) = e^{-\lambda}\frac{\lambda^x}{x!}, \qquad x = 0, 1, 2, \dots ,$ mit $\lambda = 7$?
# 3. Stellen Sie Binomial- und Poissonverteilung für geeignete Werte für $n$ und $p$, in Python, dar. Verwenden Sie dazu die Funktionen `np.random.binomial(n , p , size = Anzahl der Datenpunkte)` und `poisson.rvs(\lambda = 7)` die Sie mit `from scipy.stats import poisson` importieren können. Stellen Sie die Binomialverteilung jeweils für ($n,p$) , ($10 \cdot n,\frac{p}{10}$) und ($100 \cdot n,\frac{p}{100}$) dar. Interpretieren Sie die Ergebnisse.

# -------------------------------------------------------

# ### Lösung

# **1.** Für die **Binomialverteilung** ist der Mittelwert gegeben durch $\mu = np$ und die Standardabweichung ist gleich $\sigma = np(1 - p)$.
# 
# Für die **Poissonverteilung** gilt der Mittelwert gegeben durch $\mu = \lambda$ und die Standardabweichung ist gleich $\sigma = \sqrt{\lambda}$.
# ({cite:p}`fahrmeirstatistik` s.237, s.243)

# **2.** Die Binomialverteilung approximiert die Poissonverteilung für große Werte für $n$, ($n >> 1$) und kleines $p$, ($p << 1$)

# **3.** Damit die Poissonverteilung und Binomialverteilung den gleichen Mittelwert setzen wir diese gleich und lösen nach $n$ und $p$ auf.
# 
# 1. $\lambda = 7 = \mu = np \Rightarrow \frac{7}{n} = p$
# 
# 2. $\sigma = \sqrt{\lambda}=\sqrt{7}= np(1 - p) \ \text{wir setzen} \  p \ \text{in 2. ein} \Rightarrow \sqrt{7}= 7(1 - \frac{7}{n})$
# 
# Auflösen nach $n$ ergibt $n = \frac{-7}{\frac{\sqrt{7}}{7}-1}\approx 11 \Rightarrow p \approx \frac{7}{11} \approx 0,64$
# 
# Wir erhöhen $n$ in $10$-Schritten und senken $p$ um denselben Faktor. Dadurch bleibt der Mittelwert gleich aber die Standardabweichung erhöht sich. Wir erkennen das die Binomialverteilung sich der Poissonverteilung annähert.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import poisson


# In[2]:


a = np.random.binomial(n = 11, p = 0.62, size = 1000)

bins2 = max(a)-min(a)

b = np.random.binomial(n = 110, p = 0.062, size = 1000)

bins3 = max(b)-min(b)
c = np.random.binomial(n = 1100, p = 0.0062, size = 1000)

bins4 = max(c)-min(c)


# In[3]:


y = []
for i in range(2000):
    y.append(poisson.rvs(7))
bins1 = max(y)-min(y)
x = np.linspace(0,50,500)
y_cdf  = poisson.cdf(x, 7) 

fig, ax1 = plt.subplots()
plt.xlim(-1,50)
plt.title('$\mu$ = 7')
ax1.set_xlabel('Anzahl der Ereignisse')
ax1.set_ylabel('Wahrscheinlichkeit (P=X)')
ax1.hist(y,bins1,edgecolor='red',density=True,label = 'Poissonverteilung')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  
ax2.hist(a,bins2,edgecolor='k',density=True, alpha= 1,label = 'Binomialverteilung n = 11 , p = 0,62',color = 'yellow')

fig.tight_layout()
ax1.legend()
ax2.legend(loc='lower right')
plt.show() 


# In[4]:


y = []
for i in range(2000):
    y.append(poisson.rvs(7))
bins1 = max(y)-min(y)
x = np.linspace(0,50,500)
y_cdf  = poisson.cdf(x, 7) 

fig, ax1 = plt.subplots()
plt.xlim(-1,50)
plt.title('$\mu$ = 7')
ax1.set_xlabel('Anzahl der Ereignisse')
ax1.set_ylabel('Wahrscheinlichkeit (P=X)')
ax1.hist(y,bins1,edgecolor='red',density=True,label = 'Poissonverteilung')
ax1.tick_params(axis='y')

  
ax3 = ax1.twinx()
ax3.hist(b,bins3,edgecolor='k',density=True, alpha= 0.75,label = 'Binomialverteilung n = 110 , p = 0,062',color = 'yellow')




fig.tight_layout()
ax1.legend()
ax3.legend(loc='lower right')
plt.show() 


# In[5]:


y = []
for i in range(2000):
    y.append(poisson.rvs(7))
bins1 = max(y)-min(y)
x = np.linspace(0,50,500)
y_cdf  = poisson.cdf(x, 7) 

fig, ax1 = plt.subplots()
plt.xlim(-1,50)
plt.title('$\mu$ = 7')
ax1.set_xlabel('Anzahl der Ereignisse')
ax1.set_ylabel('Wahrscheinlichkeit (P=X)')
ax1.hist(y,bins1,edgecolor='red',density=True,label = 'Poissonverteilung')
ax1.tick_params(axis='y')

ax4 = ax1.twinx()


ax4.hist(c,bins4,edgecolor='k',density=True, alpha= 0.75,label = 'Binomialverteilung n = 1100 , p = 0,0062',color = 'yellow')


fig.tight_layout()
ax1.legend()
ax4.legend(loc='lower right')
plt.show() 

