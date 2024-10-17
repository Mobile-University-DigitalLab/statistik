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


# # Hypothesentest - unabhängige Stichproben, $\sigma_1 \approx \sigma_2$

# Bei $2$ Stichproben aus $2$ Grundgesamtheiten erhalten Sie folgende Mittelwerte und Standardabweichungen: $\mu_1=61$, $\sigma_1=15,5$, $n_1=15$, $\mu_2=48,4$ $\sigma_2=18,1$, $n_2 =12$ . Welchen Hypothesentest müssen Sie anwenden um zu prüfen ob $\mu_1 \gt \mu_2$ gilt?
# 
# 1. Formulieren Sie die geeignete Null- und Alternativhypothese.
# 
# 2. Berechnen Sie die Teststatistik. 
# 
# 
# 3. Berechnen Sie den kritischen Wert (entweder mit Python oder Wahrscheinlichkeitstabelle) bei einem Signifikanzniveau $\alpha = 0,01$. Wird $H_0$ abgelehnt? Hierbei ergint sich der kritische Wert bei $\alpha = 0,01$ für einen rechtseitigen Test $\mu_1 \gt \mu_2$ mit der $t$-Verteilung ($1- \alpha, t_{df}$).
# 
# 4. Interpretieren Sie das Ergebnis
# 
# Die gewichtete Standardabweichung ergibt sich zu:
# 
# $$s_g = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2 }{n_1+n_2-2}}\text{,}$$
# 
# Die Teststatistik ist $t$-verteilt und ergibt sich für $(\mu_1-\mu_2)=0$ zu:
# 
# $$t =  \frac{(\bar x_1 - \bar x_2)}{s_g \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

# -------------------------------------------------------

# In[2]:


# Frage 3 ...


# In[3]:


# Frage 4 ...


# ### Lösung

# ```{togle}
# $2$-Stichproben $t$-Test mit (gepoolter) gewichteter Standardabweichung.
# ```

# ```{toggle}
# $$H_0 : \mu_1 = \mu_2$$
# $$H_A : \mu_1 \gt \mu_2$$
# ```

# In[4]:


import numpy as np

n_1 = 15
n_2 = 12
mu_1 = 61
mu_2 = 48.4
s_1 = 15.5
s_2 = 18.1

s_g = np.sqrt(((n_1 - 1) * s_1**2 + (n_2 - 1) * s_2**2) / (n_1 + n_2 - 2))
t_stat = (mu_1 - mu_2) / (s_g * np.sqrt(1 / n_1 + 1 / n_2))
t_stat


# In[5]:


from scipy.stats import t

alpha = 0.01
df = 25
critical = t.ppf(1 - alpha, df=df)
print(f"Kritischer Wert: {critical}")
t_stat >= critical


# ```{toggle}
# Die Teststatistik liegt nicht im Ablehnungsbereich daher ist für $\alpha = 0,01$ keine signifikante Abweichung der Mittelwerte feststellbar. Die Nullhypothese wird daher beibehalten.
# ```

# In[ ]:




