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


# # Übung zu Pandas Dataframes

# 1. Erstellen Sie einen Pandas Dataframe (`df`) wie vorgegeben.
# 2. Fügen Sie die Spalte `nc_score` $= [2.5, 3, 2.2, 1.0] $ zum Dataframe `df` hinzu.
# 3. Wählen Sie alle Daten aus, für die gilt `age` $\gt 23 $ und speichern Sie die Werte in den neuen Dataframe `df2`.

# |id|Name|Age|
# |---|---|---|
# |1|John|26|
# |2|Alice|20|
# |3|Mike|21|
# |4|Anne|25|

# -------------------------------------------------------

# In[2]:


# Frage 1 ...


# In[3]:


# Frage 2 ...


# In[4]:


# Frage 3 ...


# ## Lösungen

# In[5]:


import pandas as pd

df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "Name": ["John", "Alice", "Mike", "Anne"],
        "Age": [26, 20, 21, 25],
    }
)
df


# In[6]:


df["nc_score"] = [2.5, 3, 2.2, 1.0]
df


# In[7]:


df2 = df.loc[df["Age"] < 23]
df2

