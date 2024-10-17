#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 2
# ### Aufgabenstellung 6 : Multiple Choice Test
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Sie entscheiden sich bei einem Multiple Choice Test mit $10$ Fragen mit jeweils $4$ Antwortmöglichkeiten auf gut Glück zu antworten.
# 
# 1. Bestimmen Sie die Wahrscheinlichkeitsfunktion von $X$ und berechnen Sie den Erwartungswert.
# 2. Mit welcher Wahrscheinlichkeit machen Sie $2$ Fehler oder weniger?
# 3. Mit welcher Wahrscheinlichkeit bestehen Sie den Test wenn $5$ oder mehr richtige Antworten für eine positive Bewertung benötigt werden?
# 

# -------------------------------------------------------

# ### Lösung

# Wahrscheinlichkeit für richtig : $P(X)= 0,25$ 
# 
# Wahrscheinlichkeit für falsch : $1-P(X)= 0,75$
# 
# Für $X$ gilt die Binomialverteilung : $P(X = x) = {n \choose x}p^x(1 - p)^{n-x}, \qquad x = 0, 1, 2, \dots , n$ 
# 
# Der Erwartungswert ist gegeben durch $E(X) = \sum_{i=1}^{N}x_iP(X=x_i)$

# **1.**

# In[1]:


import math

n = 10
p = 0.25

prob_total =  0
for i in range(11):
    prob = math.comb(n,i)*p**(i)*(1-p)**(n-i)*i
    prob_total += prob
    
print('Der Erwartungswert zufälliger richtiger Antworten entspricht',prob_total)


# **2.** $n-k$ kann die Werte $0,1,2$ annehmen also müssen wir die Wahrscheinlichkeit für $k=8,9,10$ berechnen.

# In[2]:


n = 10
p = 0.25

prob_total =  0
for i in range(8,11):
    prob = math.comb(n,i)*p**(i)*(1-p)**(n-i)
    prob_total += prob
    
print('Die Wahrscheinlichkeit 2 oder weniger Fehler zu machen beträgt ',prob_total)


# **3.** $n-k$ kann die Werte $0,1,2,3,4,5$ annehmen also müssen wir die Wahrscheinlichkeit für $k=5,6,7,8,9,10$ berechnen.

# In[3]:


n = 10
p = 0.25

prob_total =  0
for i in range(5,11):
    prob = math.comb(n,i)*p**(i)*(1-p)**(n-i)
    prob_total += prob
    
print('Die Wahrscheinlichkeit 5 oder weniger Fehler zu machen beträgt ',prob_total)

