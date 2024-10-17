#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 7
# ### Aufgabenstellung 25 : Einfaktorielle ANOVA
# 
# -------------------------------------------------------
# -------------------------------------------------------

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


# - Führen Sie jeweils eine schrittweise einfaktorielle Varianzanalyse für die folgenden Daten durch:
# `sample_dat1,sample_dat2,sample_dat3` und `sample_dat4,sample_dat5,sample_dat6`

# In[2]:


sample_dat1 = norm.rvs(loc=0,scale=1,size = 25,random_state = 1)
sample_dat2 = norm.rvs(loc=0.01,scale=1.1,size = 30,random_state = 1)
sample_dat3 = norm.rvs(loc=-0.01,scale=1.1,size = 28,random_state = 1)
sample_dat4 = t.rvs(df = 33 , loc = 0.8 , scale = 0.8, size = 34,random_state = 1)
sample_dat5 = t.rvs(df = 26 , loc = 0.5 , scale = 1.22, size = 27,random_state = 1)
sample_dat6 = norm.rvs(loc=0,scale=1,size = 25,random_state = 1)


# 
# 
# 
# 
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
# - Benutzen Sie für Schritte $3$ und $4$ die Funktion `f_oneway()` die Sie mit `from scipy.stats import f_oneway` importieren können 
# 
# - Prüfen Sie ob Normalitätsbedingung und Verhältnis der Standardabweichungen $\lt 2$ gilt.
# 

# -------------------------------------------------------

# ### Lösung 

# In[3]:


# Erzeuge Q-Q Plot
import numpy as np 
 
import scipy.stats as stats

measurements = sample_dat1 
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()
measurements = sample_dat2 
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()
measurements = sample_dat3 
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()
measurements = sample_dat4 
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()
measurements = sample_dat5 
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()
measurements = sample_dat6 
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[4]:


std_set1 = []
std_set1.append(np.std(sample_dat1))
std_set1.append(np.std(sample_dat2))
std_set1.append(np.std(sample_dat3))
max(std_set1)/min(std_set1)


# In[5]:


std_set2 = []
std_set2.append(np.std(sample_dat4))
std_set2.append(np.std(sample_dat5))
std_set2.append(np.std(sample_dat6))
max(std_set2)/min(std_set2)


# **Schritt 1 : Geben Sie die Nullhypothese $H_0$ und alternative Hypothese $H_A$ an**

# $$H_0: \quad \mu_1=\mu_2=\mu_3$$

# **Alternative Hypothese**

# $$H_A: \quad\text{Nicht alle Mittelwerte sind gleich}$$

# **Schritt 2: Legen Sie das Signifikanzniveau,$\alpha$ fest**

# $$\alpha = 0,01$$

# In[6]:


alpha = 0.01


# **Schritt 3 und 4: Berechnen Sie den Wert der Teststatistik und den $p$-Wert**

# In[7]:


statistics, pvalue1 = f_oneway(sample_dat1,sample_dat2,sample_dat3)

print('Wert der F-Statistik :',statistics)
print('p-Wert :',pvalue1)


# In[8]:


statistics, pvalue2 = stats.f_oneway(sample_dat4,sample_dat5,sample_dat6)

print('Wert der F-Statistik :',statistics)
print('p-Wert :',pvalue2)


# **Schritt 5: Wenn $p \le \alpha , H_0$ ablehnen; ansonsten $H_0$ nicht ablehnen**

# In[9]:


pvalue1< alpha


# In[10]:


pvalue2< alpha


# **Schritt 6: Interpretieren Sie das Ergebnis des Hypothesentests**

# Im ersten Fall (`sample_dat1,sample_dat2,sample_dat3`) ergibt die ANOVA keinen signifikanten Unterschied. Wir nehmen die Null-hypothese an.

# Im zweiten Fall (`sample_dat4,sample_dat5,sample_dat6`) ergibt die ANOVA einen signifikanten Unterschied. Wir lehnen die Null-hypothese ab.

# In[ ]:




