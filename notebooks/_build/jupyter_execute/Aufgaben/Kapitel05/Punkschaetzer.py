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


# # Punktschätzungen bei unbekanntem $\sigma$

# 1. Welche Wahrscheinlichkeitsverteilung wird für die Berechnungen von Punktschätzungen bei unbekannter Standardabweichnung der Grundgesamtheit und kleiner Stichprobengrösse ($n\lt 30$) verwendet?
# 2. Wir simulieren eine Grundgesamtheit an Daten indem wir $100$ Zufallszahlen zwischen $-100$ und $100$ generieren. Berechnen Sie Mittelwert und Standardabweichung für diese Zufallsdaten. 
# 3. Nehmen Sie eine Stichprobe vom Umfang $n=10$. Berechnen Sie Mittelwert und Konfidenzintervall ($\alpha = 0,05$) für die Stichprobe unter der Verwendung einer geeigneten Wahrscheinlichkeitsverteilung und überprüfen Sie ob der Mittelwert der Grundgesamtheit innerhalb des Konfidenzintervalls liegt.

# -------------------------------------------------------

# In[2]:


# Frage 2 ...


# In[3]:


# Frage 3 ...


# ## Lösungen

# ```{toggle}
# Im Fall unbekannter Standardabweichung der Grundgesamtheit $\sigma$ kann die Standardabweichung der Stichprobe, $s$, als Schätzer verwendet werden.
# $s$ ist gegeben durch $s =\sqrt{\sum_i \frac{x_i - \bar x}{n-1}}$ und ihre Verteilung wird durch die $t$-Verteilung beschrieben.
# 
# ({cite:p}`fahrmeirstatistik` s.360)
# ```

# In[4]:


import numpy as np

# Erzeuge Random seed
np.random.seed(1)
data = []
# Generiere Zufallszahlen
np.random.seed(42)
data = [np.random.randint(-100, 100) for x in range(100)]

# Berechne Mittelwert und Standardabweichung
data_mean = np.mean(data)
print(f"Mittelwert: {data_mean}")
data_std = np.std(data)
print(f"Standardabweichung: {data_std}")


# In[5]:


import random
from scipy.stats import t

# Nehme Stichprobe
n = 10
np.random.seed(42)
sample = random.sample(data, n)

# Berechne Mittelwert und Standardabweichung der Stichprobe
sample_mean = np.mean(sample)
print(f"Mittelwert der Stichprobe: {sample_mean}")
sample_std = np.std(sample)
print(f"Standardabweichung der Stichprobe: {sample_mean}")

alpha = 0.05
lower = data_mean - t.ppf(1 - alpha / 2, df=n - 1) * (sample_std / np.sqrt(n))
upper = data_mean + t.ppf(1 - alpha / 2, df=n - 1) * (sample_std / np.sqrt(n))
print(
    f"Lower: {round(lower,3)} <= Sample mean: {sample_mean} <= Upper: {round(upper,3)}"
)

assert lower <= data_mean and data_mean <= upper

