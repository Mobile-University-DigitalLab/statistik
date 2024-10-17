#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 2
# ### Aufgabenstellung 9 : Erwartungswert
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Erklären Sie den Unterschied zwischen relativen und absoluten Häufigkeiten. 
# 2. Berechnen Sie den Erwartungswert für die Würfelsumme bei $2$ Würfen. 
# 3. Berechnen Sie in Python für $n=10,100,10000$ simulierte Würfe das arithmetische Mittel. Gegen welchen Wert konvergiert das arithmetische Mittel?

# -------------------------------------------------------

# ### Lösung

# **1.**

# **Absolute Häufigkeiten** $h(a_j)=h_j$ entsprechen der Anzahl der Ereignisse/Elemente/Messergebnisse $x_i$ für die gilt $x_i = a_j$ 
# **Relative Häufigkeiten**  entsprechen den **absoluten Häufigkeiten** geteilt durch die Gesamtanzahl $n$ betrachteten Ereignisse/Elemente/Messergebnisse $f(a_j)=f_j=\frac{h_j}{n}$
# ({cite:p}`fahrmeirstatistik` s.30)

# **2.**
# 
# Der Erwartungswert ist gegeben durch:
# 
# $$E(X) = \sum_i P(X_i)X_i$$

# In[1]:


1/36*(2+3*2+4*3+5*4+6*5+7*6+8*5+9*4+10*3+11*2+12)


# In[2]:


dices = [11,101,10001]
import random
n=12

for dice in dices:
    e_x = 0
    for i in range(dice):
        e_x += random.uniform(1,6) + random.uniform(1,6)
    print(e_x/dice,': Erwartungswert bei',i,'Würfen')


# In[ ]:




