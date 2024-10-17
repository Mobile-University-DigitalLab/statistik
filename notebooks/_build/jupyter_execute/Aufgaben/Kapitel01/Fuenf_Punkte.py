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


# # Maße der zentralen Tendenz, Streumaße und Fünf-Punkte Zusammenfassung

# 1. Berechnen Sie Mittelwert, Standardabweichung, Modalwert des Vektors `data`.
# 
# `data = [10, 1, 17, 0, 14, 2, 11, 1, 4, 10, 5, 1, 1, 99, 47, 16, 3, 4, 9, 11]`
# 
# 2. Berechnen Sie die Fünf-Punkte-Zusammenfassung für `data`.
# 
# 3. Generieren Sie mit der Funktion `np.linspace()` eine $x$-Achse mit Werten zwischen $0$ und $1$ und stellen Sie `data` als Streudiagramm dar.
# 
# 4. Stellen Sie die Daten als Boxplot dar.
# 
# 5. Bewerten Sie anhand der Ergebnisse mögliche Ausreißer.

# -------------------------------------------------------

# In[2]:


# Frage 1 ...


# In[3]:


# Frage 2 ...


# In[4]:


# Frage 3 ...


# In[5]:


# Frage 4 ...


# In[6]:


# Frage 5 ...


# ## Lösungen

# In[7]:


import numpy as np
import statistics as st

data = [10, 1, 17, 0, 14, 2, 11, 1, 4, 10, 5, 1, 1, 99, 47, 16, 3, 4, 9, 11]
print(f"Mittelwert: {np.mean(data)}")
print(f"Standardabweichung: {np.std(data)}")
print(f"Modalwert: {st.mode(data)}")


# In[8]:


# Berechne Fünf-Punkte-Zusammenfassung
# Berechne die Quartilen
q1, median, q3 = np.percentile(data, [25, 50, 75])

# Berechne minimal/maximal Datenpunkte
data_min, data_max = min(data), max(data)

# Ausgabe der Daten
print(f"Min:    {data_min}")
print(f"Q1:     {q1}")
print(f"Median: {median}")
print(f"Q3:     {q3}")
print(f"Max:    {data_max}")


# In[9]:


import matplotlib.pyplot as plt

x = np.linspace(0, 1, len(data))
fig, ax = plt.subplots()
ax.scatter(x, data, marker="o")
plt.show()


# In[10]:


fig, ax = plt.subplots()
ax.boxplot(data)
plt.show()


# ```{toggle}
# Anhand der visuellen Auswertung von Boxplot und Streudiagram liegt es nahe, dass die Datenpunkte 99 und 27 Ausreißer sind.
# ```
