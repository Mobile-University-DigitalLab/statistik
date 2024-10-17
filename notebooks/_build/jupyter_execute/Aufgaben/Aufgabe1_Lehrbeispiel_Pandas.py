#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 1
# ### Aufgabenstellung 1 : Lehrbeispiel Pandas Dataframes
# 
# -------------------------------------------------------
# -------------------------------------------------------

# In dem folgenden Lehrbeispiel diskutieren wir grundsätzliche Methoden für den Umgang mit Dataframes.

# 1. Dataframes können mit der Funktion `pd.DataFrame()` erstellt werden wobei `pd` der Aufruf für das `Pandas` Paket ist. Um Spalten hinzuzufügen verwenden wir den Syntax: `DataFrame('Spaltenname 1':[a1,b1,c1,...], 'Spaltenname 2':[a2,b2,c2,...], ...)`

# In[1]:


import pandas as pd

# Erstelle Dataframe df
df = pd.DataFrame({'numbers': [1, 2, 3], 'colors': ['red', 'white', 'blue'], 'frequency': [220, 440, 880]})
df


# 2. Wir können auch alternativ einen leeren Dataframe erstellen und die Daten Spaltenweise hinzufügen.

# In[2]:


df = pd.DataFrame()
df['numbers'] = [1, 2, 3]
df['colors'] = ['red', 'white', 'blue']
df['frequency'] = [220, 440, 880]
df


# 3. Die erste Spalte wird Index genannt. Wir können mit der Methode `loc[Index]` Zeilenweise Elemente auswählen.

# In[3]:


print('Erste Zeile von df')
print(df.loc[0])
print('Zweite Zeile von df')
print(df.loc[1])
print('Dritte Zeile von df')
print(df.loc[2])


# 4. In ähnlicher Weise können wir `loc()` verwenden, wenn wir Spalten anhand ihrer Namen auswählen wollen

# In[4]:


print(df.loc[:,['numbers','frequency']])


# 5. Wir können auch die Methoden `iloc()` und `loc()` verwenden, um mehrere Spalten auszuwählen.
#     Wenn wir die Spaltenindizes verwenden wollen, um sie zu extrahieren, können wir `iloc()` verwenden, wie im folgenden Beispiel gezeigt:

# In[5]:


print(df.iloc[[0],[0,2]])


# 5. Man kann so auch einzelne Elemente auswählen.

# In[6]:


print(df.iloc[[2],[0]])


# 6. ... oder überschreiben.

# In[7]:


df.iloc[[2],[0]] = 5
print(df.iloc[[2],[0]])


# 7. Es ist auch möglich mit `loc()` Daten zu Filtern indem wir logische Verknüpfungen verwenden

# In[8]:


df2 = df.loc[df['frequency'] < 441]
df2


# In[9]:


df2 = df.loc[df['colors'] == 'red']
df2


# 8. Man kann logische Verknüpfungen auch kombinieren um spezifischer zu filtern

# In[10]:


df2 = df.loc[(df['numbers'] >= 1) & (df['frequency'] < 441)]
df2


# In[ ]:




