#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 2
# ### Aufgabenstellung 7 : Fakultät berechnen
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Schreiben Sie eine Funktion, die die Fakultät einer Zahl zurückgibt. Und geben Sie $1!$ bis $7!$ aus. 
# 
# Lösungen können iterativ oder rekursiv sein.
# 
# Die Unterstützung für das Abfangen negativer $n$ ist optional.

# -------------------------------------------------------

# ### Lösung

# **1. Iterative Lösung**

# In[1]:


def factorial1(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result


# In[2]:


for i in range(1,8):
    print(factorial1(i))


# **oder**

# In[3]:


def factorial2(n):
    for i in range(1, n):
        n *= i
    return n


# In[4]:


for i in range(1,8):
    print(factorial2(i))


# **2. Rekursive Lösung**

# In[5]:


def factorial3(n):
    z=1
    if n>1:
        z=n*factorial3(n-1)
    return z


# In[6]:


for i in range(1,8):    
    print(factorial3(i))


# **oder**

# In[7]:


def factorial4(n):
    return n * factorial4(n - 1) if n else 1


# In[8]:


for i in range(1,8):
    print(factorial4(i))


# In[ ]:




