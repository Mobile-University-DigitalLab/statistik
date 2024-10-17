#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 6
# ### Aufgabenstellung 23 : $2$-Stichproben $t$-Test
# 
# -------------------------------------------------------
# -------------------------------------------------------

# Führen Sie einen $2$-Stichproben $t$-Test für folgende unabhängige Daten mit gleicher Standardabweichung bei einem Signifikanzniveau $\alpha = 0,01$ aus:

# In[1]:


from scipy.stats import norm
from scipy.stats import t
a = norm.rvs(loc=0, scale=2, size = 100,random_state = 1)
b = norm.rvs(loc=1, scale=2, size = 100,random_state = 1)


# Überprüfen Sie die Nullhypothese:
# 
# $$H_0: \quad \mu_1 = \mu_2$$
# 
# und 
# 
# alternative Hypothese:
# 
# $$H_A: \quad \mu_1 \ne \mu_2$$
# 
# 1. für den $p$-Wert Ansatz
# 
# 2. für den kritischen Wert

# -------------------------------------------------------

# ### Lösung

# ### Überprüfung der Hypothesen

# Wir führen einen $2$-Stichproben $t$-Test durch, indem wir das schrittweise Durchführungsverfahren für Hypothesentests befolgen.

# **Schritt 1 : Geben Sie die Nullhypothese $H_0$ und alternative Hypothese $H_A$ an**

# Die Nullhypothese besagt, dass der Mittelwert des $1$-ten Datensatzes ($μ_1$) gleich dem Mittelwert des $2$-ten Datensatzes ($μ_2$) ist.

# $$H_0: \quad \mu_1 = \mu_2$$

# Wir wollen prüfen, ob sich der Mittelwert des $1$-ten Datensatzes ($μ_1$) von dem Mittelwert des $2$-ten Datensatzes ($μ_2$) unterscheidet, daher wird die Alternativhypothese wie folgt formuliert

# **Alternative Hypothese**

# $$H_A: \quad \mu_1 \ne \mu_2$$

# Aus dieser Formulierung ergibt sich ein zweiseitiger Hypothesentest.

# **Schritt 2: Legen Sie das Signifikanzniveau,$\alpha$ fest**

# $$\alpha = 0,01$$

# In[2]:


alpha = 0.01


# **1.**

# **Schritt 3 und 4: Berechnen Sie den Wert der Teststatistik und den $p$-Wert**

# In[3]:


from scipy import stats
statistics, pvalue = stats.ttest_ind(a, b, equal_var=True)


# **Schritt 5: Wenn $p \le \alpha , H_0$ ablehnen; ansonsten $H_0$ nicht ablehnen**

# In[4]:


pvalue <= alpha


# In[5]:


pvalue


# Der $p$-Wert ist kleiner als das angegebene Signifikanzniveau von $0,01$; wir verwerfen $H_0$. Die Testergebnisse sind statistisch signifikant auf dem $1 \%$-Niveau und liefern einen sehr starken Beweis gegen die Nullhypothese.

# **Schritt 6: Interpretieren Sie das Ergebnis des Hypothesentests**

# $p=9,888 \cdot 10^{-5}$; Bei einem Signifikanzniveau von $1 \%$ lassen die Daten den Schluss zu, dass der Mittelwert des $1$-ten Datensatzes ($μ_1$) nicht gleich dem Mittelwert des $2$-ten Datensatzes ($μ_2$) ist.

# **2.**

# **Schritt 3 und 4: Berechnen Sie den Wert der Teststatistik und den kritischen Wert**

# In[6]:


statistics, pvalue = stats.ttest_ind(a, b, equal_var=True)


# In[7]:


# unterer kritischer Punkt
lower = t.ppf(alpha/2,df = 99)


# In[8]:


# oberer kritischer Punkt
upper = t.ppf(1-alpha/2,df = 99)


# **Schritt 5: Wenn gilt: Teststatistik $\lt $ unterer kritischer Wert  oder Teststatistik $\gt$ oberer kritischer Wert, $H_0$ ablehnen; ansonsten $H_0$ nicht ablehnen**

# In[9]:


statistics <= lower 


# In[10]:


statistics >= upper 


# Aufgrund der numerischen Auswertung fällt der Wert in den Verwerfungsbereich, so dass wir $H_0$ verwerfen. Die Testergebnisse sind auf dem $1 \%$-Niveau statistisch signifikant.

# In[ ]:




