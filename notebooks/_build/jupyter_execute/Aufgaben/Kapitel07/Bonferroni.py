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


# # Bonferroni Korrektur

# 1. Führen Sie einen post-hoc Mehrfachhypothesentests mit den Datensätzen `sample_dat1`, `sample_dat2` und `sample_dat3` durch um zu bestimmen welcher Datensatz sich von den anderen unterscheidet. Berechnen Sie hierfür die Bonferroni Korrektur bei einen $\alpha=0.05$.
# Die Bonferroni Korrektur ergibt sich zu:
# 
# $$\alpha = \frac{\alpha}{m}\text{,}$$
# 
# $$m=\frac{k(k-1)}{2}\text{,}$$
# 
# $$\alpha = \frac{0,05}{3}$$
# 
# 2. Interpretieren Sie das Ergebnos

# In[2]:


from scipy.stats import norm, t

rs = 1
sample_dat1 = t.rvs(df=33, loc=0.8, scale=0.8, size=34, random_state=rs)
sample_dat2 = t.rvs(df=26, loc=0.5, scale=1.22, size=27, random_state=rs)
sample_dat3 = norm.rvs(loc=0, scale=1, size=25, random_state=rs)


# -------------------------------------------------------

# In[3]:


# Frage 1 ...


# ## Lösungen

# In[4]:


from scipy.stats import ttest_ind

alpha = 0.05
k = 3
m = k * (k - 1) / 2
bonf = alpha / m
print(f"Bonferroni Korrektur: {bonf}")

combinations = [
    (sample_dat1, sample_dat2),
    (sample_dat1, sample_dat3),
    (sample_dat2, sample_dat3),
]

for combination in combinations:
    statistics, p_value = ttest_ind(combination[0], combination[1])
    print(f"Reject H0: {p_value <= bonf}")


# ```{toggle}
# Der Mehrfachvergleich ergibt für `sample_dat1` und `sample_dat3` einen signifikanten Unterschiede bei Bonferroni korrigiertem $\alpha_B = \frac{0,05}{3}$.
# ```
