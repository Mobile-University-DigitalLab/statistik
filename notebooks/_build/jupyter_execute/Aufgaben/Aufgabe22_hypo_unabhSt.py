#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 6
# ### Aufgabenstellung 22 : Hypothesentest - unabhängige Stichproben, $\sigma_1 \approx \sigma_2$
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Bei $2$ Stichproben aus $2$ Grundgesamtheiten erhalten Sie folgende Mittelwerte und Standardabweichungen: $\mu_1=61$, $\sigma_1=15,5$ $\mu_2=48,4$ $\sigma_2=18,1$. Welchen Hypothesentest müssen Sie anwenden um zu prüfen ob $\mu_1 \gt \mu_2$ gilt?
# 
# 2. Formulieren Sie die geeignete Null- und Alternativhypothese.
# 
# 3. Berechnen Sie die Teststatistik.
# 
# 4. Berechnen Sie den kritischen Wert (entweder mit Python oder Wahrscheinlichkeitstabelle) bei einem Signifikanzniveau $\alpha = 0,01$. Wird $H_0$ abgelehnt ?

# -------------------------------------------------------

# ### Lösung

# **1.**
# 
# $2$-Stichproben $t$-Test mit (gepoolter) gewichteter Standardabweichung.

# **2.**
# 
# $$H_0 : \mu_1 = \mu_2$$
# 
# $$H_A : \mu_1 \gt \mu_2$$

# **3.**
# 
# Die gewichtete Standardabweichung ergibt sich zu:
# 
# $$s_g = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2 }{n_1+n_2-2}}\text{,}$$

# In[1]:


import numpy as np
n_1 = 15
n_2 = 12
mu_1 = 61
mu_2 = 48.4
s_1 = 15.5
s_2 = 18.1

s_g = np.sqrt(((n_1-1)*s_1**2+(n_2-1)*s_2**2)/(n_1+n_2-2))
s_g


# Die Teststatistik ist $t$-verteilt und ergibt sich für $(\mu_1-\mu_2)=0$ zu:
# 
# $$t =  \frac{(\bar x_1 - \bar x_2)}{s_g \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

# In[2]:


t_stat = (mu_1-mu_2)/(s_g*np.sqrt(1/n_1+1/n_2))
t_stat


# **4.**
# 
# Der kritische Wert bei $\alpha = 0,01$ für einen rechtseitigen Test $\mu_1 \gt \mu_2$ mit der $t$-Verteilung ($1- \alpha, t(25)$) zu:

# In[3]:


from scipy.stats import t
alpha = 0.01
df = 25
critical = t.ppf(1-alpha,df=25)
critical


# In[4]:


t_stat >= critical 


# Die Teststatistik liegt nicht im Ablehnungsbereich daher ist für $\alpha = 0,01$ keine signifikante Abweichung der Mittelwerte feststellbar. Die Nullhypothese wird daher beibehalten.

# In[ ]:




