#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 1
# ### Aufgabenstellung 5 : Median berechnen
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Schreiben Sie eine Funktion die den Median einer Liste / eines Arrays an Daten ausgibt. Das Programm soll sowohl bei gerader wie ungerader Anzahl von Elementen das richtige Ergebnis liefern. Benutzen Sie dazu den arithmetischen Operator `Floor division`. Mehr zu Python Opratoren finden Sie 
# <a href="https://www.w3schools.com/python/python_operators.asp">hier</a>.
# Berechnen Sie den Median für die Daten:

# In[1]:


a = (4.1, 5.6, 7.2, 1.7, 9.3, 4.4, 3.2)
b = (4.1, 7.2, 1.7, 9.3, 4.4, 3.2)


# -------------------------------------------------------

# ### Lösung

# In[2]:


def median(array):
    sortd = sorted(array)
    alen = len(sortd)
    return 0.5*( sortd[(alen-1)//2] + sortd[alen//2])
 
print('Datensatz :',a, 'Median :', median(a))
print('Datensatz :',b, 'Median :', median(b))


# In[ ]:




