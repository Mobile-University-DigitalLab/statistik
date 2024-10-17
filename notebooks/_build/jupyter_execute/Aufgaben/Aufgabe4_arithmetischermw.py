#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 1
# ### Aufgabenstellung 4 : Arithmetischen Mittelwert berechnen
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Schreiben sie eine Funktion die den Mittelwert einer Liste/ eines Arrays berechnet.
# Verwenden Sie dazu die folgenden Daten:

# In[1]:


a = [1,2,3,4,5,6,7,8,9,10]
b = (-10,-8,-6,-4,-2,0)


# -------------------------------------------------------

# In[2]:


def average_arithmic(values):
    return sum(values)/len(values)


# In[3]:


average_arithmic(a)


# In[4]:


average_arithmic(b)

