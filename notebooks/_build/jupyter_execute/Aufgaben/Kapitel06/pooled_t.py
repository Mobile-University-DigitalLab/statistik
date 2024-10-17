#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Load the "autoreload" extension
get_ipython().run_line_magic('load_ext', 'autoreload')
# always reload modules
get_ipython().run_line_magic('autoreload', '2')
# black formatter for jupyter notebooks
# %load_ext nb_black
# black formatter for jupyter lab
get_ipython().run_line_magic('load_ext', 'lab_black')

get_ipython().run_line_magic('run', '../../../src/notebook_env.py')


# # $2$-Stichproben $t$-Test

# Führen Sie einen $2$-Stichproben $t$-Test für folgende unabhängige Daten mit gleicher Standardabweichung bei einem Signifikanzniveau $\alpha = 0,01$ aus:

# In[2]:


from scipy.stats import norm

n = 100
a = norm.rvs(loc=0, scale=2, size=n, random_state=1)
b = norm.rvs(loc=1, scale=2, size=n, random_state=1)


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
# 
# 3. Interpretieren Sie das Ergebnis

# -------------------------------------------------------

# ## Lösungen

# In[3]:


from scipy import stats

alpha = 0.01
statistics, pvalue = stats.ttest_ind(a, b, equal_var=True)
pvalue <= alpha


# In[4]:


statistics, pvalue = stats.ttest_ind(a, b, equal_var=True)
# unterer kritischer Punkt
lower = stats.t.ppf(alpha / 2, df=n - 1)
# oberer kritischer Punkt
upper = stats.t.ppf(1 - alpha / 2, df=n - 1)
(statistics <= lower) or (statistics >= upper)


# ```{toggle}
# Wir führen einen $2$-Stichproben $t$-Test durch.
# 
# Die Nullhypothese besagt, dass der Mittelwert des $1$-ten Datensatzes ($μ_1$) gleich dem Mittelwert des $2$-ten Datensatzes ($μ_2$) ist.
# $$H_0: \quad \mu_1 = \mu_2$$
# 
# Wir wollen prüfen, ob sich der Mittelwert des $1$-ten Datensatzes ($μ_1$) von dem Mittelwert des $2$-ten Datensatzes ($μ_2$) unterscheidet, daher wird die Alternativhypothese wie folgt formuliert:
# $$H_A: \quad \mu_1 \ne \mu_2$$
# 
# Aus dieser Formulierung ergibt sich ein zweiseitiger Hypothesentest.
# 
# Der $p$-Wert ist kleiner als das angegebene Signifikanzniveau von $0,01$; wir verwerfen $H_0$. Die Testergebnisse sind statistisch signifikant auf dem $1 \%$-Niveau und liefern einen sehr starken Beweis gegen die Nullhypothese.
# 
# Alternaite, wenn gilt, dass die Teststatistik $\lt $ unterer dem kritischer Wert oder die Teststatistik $\gt$ oberer dem kritischer Wert liegt, müssen wir $H_0$ ablehnen.
# ```

# In[ ]:




