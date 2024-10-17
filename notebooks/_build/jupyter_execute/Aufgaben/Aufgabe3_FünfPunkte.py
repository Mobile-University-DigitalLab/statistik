#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 1
# ### Aufgabenstellung 3 : Maße der zentralen Tendenz, Streumaße und Fünf-Punkte Zusammenfassung
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Berechnen Sie Mittelwert, Standardabweichung, Modalwert für die Zahlen in `daten`.
# 
# 
# 
# `daten = [10, 1, 17, 0, 14, 2, 11, 1, 4, 10, 5, 1, 1, 99, 47, 16, 3, 4, 9, 11]`

# 2. Berechnen Sie die Fünf-Punkte-Zusammenfassung für `daten`.

# 3. Generieren Sie mit der Funktion `np.linspace(min,max,Anzahl=Datenpunkte)` eine $x$-Achse und Stellen Sie `daten` als Scatterplot dar.

# 4. Stellen Sie die Daten als Boxplot dar.

# 5. Bewerten Sie anhand der Ergebnisse mögliche Ausreißer.

# -------------------------------------------------------

# ### Lösung :

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statistics as st


# In[2]:


daten = [10, 1, 17, 0, 14, 2, 11, 1, 4, 10, 5, 1, 1, 99, 47, 16, 3, 4, 9, 11]


# 1.

# In[3]:


print('Mittelwert :', np.mean(daten))
print('Standardabweichung :', np.std(daten))
print('Modalwert :', st.mode(daten))


# 2.

# In[4]:


# Berechne Fünf-Punkte-Zusammenfassung
scores = daten

# Berechne die Quartilen
q1, median, q3 = np.percentile(scores, [25, 50, 75])

# Berechne minimal/maximal Datenpunkte
data_min, data_max = min(scores), max(scores)

# Ausgabe der Daten
print(f"Min:    {data_min}")
print(f"Q1:     {q1}")
print(f"Median: {median}")
print(f"Q3:     {q3}")
print(f"Max:    {data_max}")


# 3.

# In[5]:


x = np.linspace(0,1,20)


# In[6]:


plt.scatter(x,daten, marker="o")


# 4.

# In[7]:


plt.boxplot(daten)


# 5.

# Anhand der visuellen Auswertung von Box- und Scatterplot und der grossen Abweichung vom Mittelwert ist zu vermuten das der Datenpunkt $99$ und $27$ Ausreißer sein könnten.

# In[ ]:




