#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 6
# ### Aufgabenstellung 21 : Fehler $1$-ter und $2$-ter Art, Signifikanzniveau
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Erklären Sie was man unter Fehler $1$-ter und $2$-ter Art versteht.
# 2. Warum kann man $\alpha$ nicht beliebig genau wählen?
# 3. Eine Maschine produziert im Schnitt $2 \%$ Ausschuss. Unter $100$ Stück finden sich $5$ Defekte. Die Nullhypothese ist das die Maschine $2 \%$ Ausschuss hat. Berechnen Sie Wahrscheinlichkeit des $\alpha$-Fehlers (also bei $5$ Stück oder mehr Ausfällen fälschlicherweise die Nullhypothese abzulehnen) unter Annahme von binomialverteilten Abweichungen.

# -------------------------------------------------------

# ### Lösung

# **1.**
# 
# Der Fehler $1$-ter Art, der auch $\alpha$-Fehler oder $\alpha$ Signifikanzniveau genannt wird, kann als Wahrscheinlichkeit interpretiert werden die Nullhypothese $H_0$ fälschlicherweise abzulehnen. Meist wird bei Hypothesentests ein Signifikanzniveau von $\alpha = 0,05$ oder $\alpha = 0,01$ gewählt.
#  
# Der Fehler $2$-ter Art oder $\beta$-Fehler wird begangen, wenn die Alternativhypothese $H_0$ fälschlicherweise abgelehnt wird.

# **2.** 
# 
# Der $\alpha$-Fehler kann nicht beliebig genau gewählt werden, weil die Wahrscheinlichkeit einen Fehler $2$-ter Art zu begehen damit zunimmt. Für die meisten Experimente ist $\alpha = 0,05$ ein geeigneter Wert um Fehler der $1$-ten und $2$-ten Art zu minimieren.

# **3.**
# 
# Wir können den $\alpha$-Fehler entweder direkt ($X \le 95$) berechen:

# In[1]:


import math

n = 100
p = 0.98

prob_total =  0
for i in range(0,96):
    prob = math.comb(n,i)*p**(i)*(1-p)**(n-i)
    prob_total += prob
    
print('Die Wahrscheinlichkeit einen Fehler 1ter Art zu begehen ist',prob_total)


# oder über die Gegenwahrscheinlichkeit ($X \lt 5$)

# In[2]:


n = 100
p = 0.02

prob_total =  0
for i in range(0,5):
    prob = math.comb(n,i)*p**(i)*(1-p)**(n-i)
    prob_total += prob
    
print('Die Wahrscheinlichkeit einen Fehler 1ter Art zu begehen ist',1-prob_total)


# In[ ]:




