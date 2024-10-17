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


# # Häufigkeiten und Erwartungswert

# Betrachten wir den Würfelwurf für einen fairen sechseitigen Würfel.
# Bei $10$ Würfen werden die folgenden Zahlen gewürfelt: $1,1,2,5,6,3,4,2,4,5$
# 
# 1) Erklären Sie anhand dieser Stichprobe relative und absolute Häufigkeiten.
# 
# 2) Berechnen Sie den Erwartungswert der Stichprobe.
# 
# 3) Berechnen Sie den Erwartungswert für einen Würfel unter der Annahme von Laplacewahrscheinlichkeit (gleicher Wahrscheinlichkeit für alle $x_i$ von $X$)

# In[2]:


# Frage 1 ...


# In[3]:


# Frage 2 ...


# In[4]:


# Frage 3 ...


# ## Lösungen

# **Frage 1**
# 
# **Absolute Häufigkeiten** $h(a_j)=h_j$ entsprechen der Anzahl der Ereignisse/Elemente/Messergebnisse $x_i$ für die gilt $x_i = a_j$.
# **Relative Häufigkeiten**  entsprechen den **absoluten Häufigkeiten** geteilt durch die Gesamtanzahl $n$ betrachteten Ereignisse/Elemente/Messergebnisse $f(a_j)=f_j=\frac{h_j}{n}$
# ({cite:p}`fahrmeirstatistik` s.30).
# 
# Angewandt auf die Stichprobe ergeben sich die absoluten und relativen Häufigkeiten wie folgt:
# 
# |$X$|Absolute Häufigkeit|Relative Häufigkeit|
# |---|:---:|:---:|
# |1|2|0,2|
# |2|2|0,2|
# |3|1|0,1|
# |4|2|0,2|
# |5|2|0,2|
# |6|1|0,1|

# **Frage 2**
# 
# $$E(X) = \sum_{i=1}^6 P(x_i) x_i = 1 \cdot 0,2 + 2 \cdot 0,2 + 3 \cdot 0,1 + 4 \cdot 0,2 + 5 \cdot 0,2 + 6 \cdot 0,1 = 3,3$$

# **Frage 3**
# 
# Da von Laplacewahrscheinlichkeit ausgegangen werden kann gilt: $P(x_i)= \frac{1}{6} \approx 0,167$
# 
# $$E(X) = \sum_{i=1}^6 P(x_i) x_i = 1 \cdot 0,167 + 2 \cdot 0,167 + 3 \cdot 0,167 + 4 \cdot 0,167 + 5 \cdot 0,167 + 6 \cdot 0,167 = 3,5$$
