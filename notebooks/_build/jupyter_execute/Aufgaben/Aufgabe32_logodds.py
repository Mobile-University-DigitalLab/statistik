#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 9
# ### Aufgabenstellung 32 : Odds und Log-Odds
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Zeigen Sie, dass wenn die Odds und das einfache (binomiale) logistische Regressionsmodell ($\pi$) gegeben sind durch:
# 
# $$\text{Odds} = \frac{\pi}{1-\pi}$$ 
#  
# $$\pi = \frac{1}{1+e^{-\eta}} = \frac{1}{1+e^{-(\beta_0+ \beta_1x_1)}}$$
# 
# die Log-Odds gleich 
# $$\text{Log-Odds} =\log \biggl( \frac{\pi}{1-\pi} \biggr)= \beta_0+ \beta_1x_1$$ sind.

# -------------------------------------------------------

# ### Lösung

# Wir setzen das logische Regressionsmodell in die Odds ein:
# 
# $$ \frac{\pi}{1-\pi} = \frac{(1+e^{-(\beta_0 + \beta_1 x_1)})^{-1}}{1-(1+e^{-(\beta_0 + \beta_1 x_1)})^{-1}} = $$
# 
# $$= \frac{1}{1+e^{-(\beta_0 + \beta_1 x_1)}-1} = e^{(\beta_0 + \beta_1 x_1)}$$
# 
# $$\log \left( \frac{\pi}{1-\pi}\right) = \log (e^{(\beta_0 + \beta_1 x_1)}) = \beta_0 + \beta_1 x_1 $$

# In[ ]:





# In[ ]:




