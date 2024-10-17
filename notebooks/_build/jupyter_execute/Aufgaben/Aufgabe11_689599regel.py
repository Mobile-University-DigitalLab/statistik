#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 3
# ### Aufgabenstellung 10 : $68-95-99,7$-Regel 
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Was sind die Hauptaussagen der $68-95-99,7$-Regel?
# 2. Berechnen Sie die entsprechenden Integrale nach der $68-95-99,7$-Regel für die Normalverteilung $X \sim N( -2, 2)$

# -------------------------------------------------------

# ### Lösung

# **1.**

# Die Kernaussagen der $68-95-99,7$-Regel lauten:
#  1)  $68 \%$ der Beobachtungen liegen innerhalb einer Standardabweichung des Mittelwerts,
#  2)  $95 \%$ der Beobachtungen liegen innerhalb von zwei Standardabweichungen des Mittelwerts, und
#  3)  $99,7 \%$ der Beobachtungen liegen innerhalb von drei Standardabweichungen des Mittelwerts.
#  
#  ({cite:p}`fahrmeirstatistik` s.86)

# **2.**

# In[1]:


from scipy.stats import norm

print('1te Standardabweichung',norm.cdf(0, loc=-2, scale=2)-norm.cdf(-4, loc=-2, scale=2))
print('2te Standardabweichung',norm.cdf(2, loc=-2, scale=2)-norm.cdf(-6, loc=-2, scale=2))
print('3te Standardabweichung',norm.cdf(4, loc=-2, scale=2)-norm.cdf(-8, loc=-2, scale=2))


# In[ ]:




