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


# # Fehler $1$-ter und $2$-ter Art

# 1. Erklären Sie was man unter Fehler $1$-ter und $2$-ter Art versteht.
# 2. Warum kann man $\alpha$ nicht beliebig genau wählen?
# 3. Eine Maschine produziert im Schnitt $2 \%$ Ausschuss. Unter $100$ Stück finden sich $5$ Defekte. Die Nullhypothese ist das die Maschine $2 \%$ Ausschuss hat. Berechnen Sie Wahrscheinlichkeit des $\alpha$-Fehlers (also bei $5$ Stück oder mehr Ausfällen fälschlicherweise die Nullhypothese abzulehnen) unter Annahme von binomialverteilten Abweichungen.

# -------------------------------------------------------

# In[2]:


# Frage 3 ...


# ## Lösungen

# ```{toggle}
# Der Fehler $1$-ter Art, der auch $\alpha$-Fehler oder $\alpha$ Signifikanzniveau genannt wird, kann als Wahrscheinlichkeit interpretiert werden die Nullhypothese $H_0$ fälschlicherweise abzulehnen. Meist wird bei Hypothesentests ein Signifikanzniveau von $\alpha = 0,05$ oder $\alpha = 0,01$ gewählt.
#  
# Der Fehler $2$-ter Art oder $\beta$-Fehler wird begangen, wenn die Alternativhypothese $H_0$ fälschlicherweise abgelehnt wird.
# ```

# ```{toggle}
# Der $\alpha$-Fehler kann nicht beliebig genau gewählt werden, weil die Wahrscheinlichkeit einen Fehler $2$-ter Art zu begehen damit zunimmt. Für die meisten Experimente ist $\alpha = 0,05$ ein geeigneter Wert um Fehler der $1$-ten und $2$-ten Art zu minimieren.
# ```

# In[3]:


from scipy.stats import binom

n = 100
p = 0.98
k = 95
p = binom.cdf(k=k, p=p, n=n)
print(f"Die Wahrscheinlichkeit einen Fehler 1ter Art zu begehen ist {p}")

