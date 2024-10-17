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


# # Die 68-95-99,7-Regel 

# 1. Was sind die Hauptaussagen der $68-95-99,7$-Regel?
# 2. Berechnen Sie die entsprechenden Integrale nach der $68-95-99,7$-Regel für die Normalverteilung $X \sim N( -2, 2)$

# -------------------------------------------------------

# In[2]:


# Frage 2 ...


# ## Lösungen

# **Frage 1**
# 
# ```{toggle}
# Die Kernaussagen der $68-95-99,7$-Regel lauten:
# 1. $68 \%$ der Beobachtungen liegen innerhalb __einer__ Standardabweichung des Mittelwerts,
# 2. $95 \%$ der Beobachtungen liegen innerhalb von __zwei__ Standardabweichungen des Mittelwerts, und
# 3. $99,7 \%$ der Beobachtungen liegen innerhalb von __drei__ Standardabweichungen des Mittelwerts.
#  
#  ({cite:p}`fahrmeirstatistik` s.86)
#  ```

# **Frage 2**

# In[3]:


from scipy.stats import norm

loc = -2
scale = 2
for sd in [1, 2, 3]:
    print(
        f"{sd} Standardabweichung(en):",
        norm.cdf(loc + sd * scale, loc=loc, scale=scale)
        - norm.cdf(loc - sd * scale, loc=loc, scale=scale),
    )

