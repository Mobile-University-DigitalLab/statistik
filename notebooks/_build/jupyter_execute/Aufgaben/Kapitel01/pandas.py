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


# # Die Pandas Bibliothek

# Die <a href="https://pandas.pydata.org//">Pandas-Bibliothek</a> wurde 2010 von <a href="https://wesmckinney.com/">Wes McKinney</a> entwickelt. pandas bietet **Datenstrukturen** und **Funktionen** für die Manipulation, Verarbeitung, Bereinigung und Verwertung von Daten. Im Python-Ökosystem ist pandas das modernste Werkzeug für die Arbeit mit tabellarischen oder tabellenähnlichen Daten, bei denen jede Spalte von einem anderen Typ sein kann (`String`, `numerisch`, `Datum` oder andere). pandas bietet ausgefeilte Indizierungsfunktionen, die das Umformen, Zerlegen, Aggregieren und Auswählen von Teilmengen von Daten erleichtern. pandas stützt sich auf andere Pakete, wie <a href="https://numpy.org/">NumPy</a> und <a href="https://scipy.org/">SciPy</a>. Außerdem integriert pandas <a href="https://matplotlib.org/">matplotlib</a> zum Plotten.
# 
# Wenn Sie neu im Umgang mit pandas sind, empfehlen wir Ihnen dringend, die sehr gut geschriebenen <a href="https://pandas.pydata.org/pandas-docs/stable/getting_started/tutorials.html">pandas-Tutorials</a> zu besuchen, die alle relevanten Abschnitte für neue Benutzer abdecken, um richtig loszulegen.
# 
# Nach der Installation (Details finden Sie in der <a href="https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html">Dokumentation</a>) wird pandas mit dem kanonischen Alias `pd` importiert.

# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# Die Pandas-Bibliothek verfügt über zwei bewährte Datenstrukturen: **Series** und **DataFrame**.
# 
#   -  eindimensionales pd.Series-Objekt
#   -  zweidimensionales pd.DataFrame-Objekt

# ## Das `pd.Series` Objekt

# Erzeugung von Daten

# In[4]:


# importiere das random module von numpy
from numpy import random

# setze seed
random.seed(123)
# Erzeuge 26 Zufallszahlen zwischen -10 and 10
my_data = random.randint(low=-10, high=10, size=26)
# Ausgabe
my_data


# In[5]:


type(my_data)


# Eine Series ist ein eindimensionales Array-ähnliches Objekt, das ein Array mit Daten und ein zugehöriges Array mit Datenbeschriftungen, genannt Index, enthält. Wir erstellen ein `pd.Series-Objekt`, indem wir die Funktion `pd.Series()` aufrufen.

# In[6]:


# Entkommentieren für Dokumentation

# docstring
# ?pd.Series

# source
# ??pd.Series


# In[7]:


# Erzeuge pd.Series Objekt
s = pd.Series(data=my_data)
s


# In[8]:


type(s)


# ### `pd.Series`-Attribute

# Python-Objekte im Allgemeinen und die `pd.Series` im Besonderen bieten nützliche objektspezifische Attribute.
# 
# *Attribut* ->`OBJECT.attribute` 
# 
# *Beachten Sie, dass das Attribut ohne Klammern aufgerufen wird*

# In[9]:


s.dtypes


# In[10]:


s.index


# Wir können das Attribut `index` verwenden, um einem `pd.Series-Objekt` einen Index zuzuweisen.
# 
# Betrachten wir die Buchstaben des Alphabets....

# In[11]:


import string

letters = string.ascii_uppercase
letters


# In[12]:


s.index = list(letters)
s


# ### `pd.Series`-Methoden

# In[13]:


s.sum()


# In[14]:


s.mean()


# In[15]:


s.max()


# In[16]:


s.min()


# In[17]:


s.median()


# In[18]:


s.quantile(q=0.5)


# In[19]:


s.quantile(q=[0.25, 0.5, 0.75])


# ### Elementweise Arithmetik

# Eine sehr nützliche Eigenschaft von `pd.Series`-Objekten ist, dass wir arithmetische Operationen *elementweise* anwenden können.

# In[20]:


s + 10
# s*0.1
# 10/s
# s**2
# (2+s)*1**3
# s+s


# ### Auswahl und Indizierung

# Eine weitere wichtige Datenoperation ist die Indizierung und Auswahl bestimmter Teilmengen des Datenobjekts. pandas verfügt über einen <a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html">sehr umfangreichen Satz</a> von Methoden für diese Art von Aufgaben.
# 
# In der einfachsten Form indizieren wir eine Reihe numpy-ähnlich, indem wir den `[ ]` Operator verwenden, um einen bestimmten `Index` der Reihe auszuwählen.

# In[21]:


s


# In[22]:


s[3]


# In[23]:


s[2:6]


# In[24]:


s["C"]


# In[25]:


s["C":"K"]


# ## Das `pd.DataFrame`-Objekt

# Die primäre Datenstruktur von Pandas ist der `DataFrame`. Es handelt sich um eine zweidimensionale, größenveränderliche, potenziell heterogene tabellarische Datenstruktur mit Zeilen- und Spaltenbeschriftungen. Arithmetische Operationen richten sich sowohl auf Zeilen- als auch auf Spaltenbeschriftungen aus. Grundsätzlich kann man sich den `DataFrame` als einen `dictionary`-artigen Container für Seriesobjekte vorstellen.
# 
# **Erzeugen eines `DataFrame`-Objekts von Grund auf**
# 
# pandas erleichtert den Import verschiedener Datentypen und -quellen, aber für dieses Tutorial erzeugen wir ein DataFrame-Objekt von Grund auf.
# 
# Quelle: http://duelingdata.blogspot.de/2016/01/the-beatles.html

# In[26]:


df = pd.DataFrame(
    {
        "id": range(1, 5),
        "Name": ["John", "Paul", "George", "Ringo"],
        "Last Name": ["Lennon", "McCartney", "Harrison", "Star"],
        "dead": [True, False, True, False],
        "year_born": [1940, 1942, 1943, 1940],
        "no_of_songs": [62, 58, 24, 3],
    }
)
df


# ### `pd.DataFrame`-Attribute

# In[27]:


df.dtypes


# In[28]:


# Achse 0
df.columns


# In[29]:


# Achse 1
df.index


# ### `pd.DataFrame`-Methoden

# **Verschaffen Sie sich einen schnellen Überblick über den Datensatz**

# In[30]:


df.info()


# In[31]:


df.describe()


# In[32]:


df.describe(include="all")


# **Index in die Variable `id` ändern**

# In[33]:


df


# In[34]:


df.set_index("id")


# In[35]:


df


# Beachten Sie, dass sich nichts geändert hat!!
# 
# Aus Gründen der Speicher- und Berechnungseffizienz gibt `Pandas` eine Ansicht des Objekts zurück, keine Kopie. Wenn wir also eine dauerhafte Änderung vornehmen wollen, müssen wir das Objekt einer Variablen zuweisen/neu zuordnen:
# 
# `df = df.set_index("id") `
# 
# oder einige Methoden haben das Argument `inplace=True`:
# 
# `df.set_index("id", inplace=True)`   

# In[36]:


df = df.set_index("id")


# In[37]:


df


# **Arithmetische Methoden**

# In[38]:


df


# In[39]:


df.sum(axis=0)


# In[40]:


df.sum(axis=1)


# ### `Groupby`-Methode

# <a href="https://www.jstatsoft.org/article/view/v040i01">Hadley Wickham 2011: The Split-Apply-Combine Strategy for Data Analysis, Journal of Statistical Software, 40(1)</a>

# ![Alt-Text](../_img/split-apply-combine.svg)

# In[41]:


df


# In[42]:


df.groupby("dead")


# In[43]:


df.groupby("dead").sum()


# In[44]:


df.groupby("dead")["no_of_songs"].sum()


# In[45]:


df.groupby("dead")["no_of_songs"].mean()


# In[46]:


df.groupby("dead")["no_of_songs"].agg(["mean", "max", "min", "sum"])


# ### Familie von `apply/map`-Methoden

# - `apply` arbeitet zeilenweise (`axis=0`, Standard) / spaltenweise (`axis=1`) auf einem `DataFrame`
# - `applymap` arbeitet elementweise auf einem `DataFrame`
# - `map` arbeitet elementweise mit einer `Series`.

# In[47]:


df


# In[48]:


# (axis=0, default)
df[["Name", "Last Name"]].apply(lambda x: x.sum())


# In[49]:


# (axis=1)
df[["Name", "Last Name"]].apply(lambda x: x.sum(), axis=1)


# *... vielleicht ein nützlicherer Fall ...*

# In[50]:


df.apply(lambda x: " ".join(x[["Name", "Last Name"]]), axis=1)


# ## Auswahl und Indizierung
# 
# **Spaltenindex**

# In[51]:


df["Name"]


# In[52]:


df[["Name", "Last Name"]]


# In[53]:


df.dead


# **Zeilenindex**
# 
# Neben dem `[ ]`-Operator verfügt Pandas über weitere Indizierungsoperatoren wie `.loc[]` und `.iloc[]`, um nur einige zu nennen.
# 
#   -  `.loc[]` basiert hauptsächlich auf **Bezeichnungen**, kann aber auch mit einem booleschen Array verwendet werden.
#  -   `.iloc[]` basiert in erster Linie auf **Ganzzahlpositionen** (von $0$ bis Länge $-1$ der Achse), kann aber auch mit einem booleschen Array verwendet werden.

# In[54]:


df.head(2)


# In[55]:


df.loc[1]


# In[56]:


df.iloc[1]


# **Zeilen- und Spaltenindizes**
# 
# `df.loc[row, col]`

# In[57]:


df.loc[1, "Last Name"]


# In[58]:


df.loc[2:4, ["Name", "dead"]]


# **logisches Indizieren**

# In[59]:


df


# In[60]:


df["no_of_songs"] > 50


# In[61]:


df.loc[df["no_of_songs"] > 50]


# In[62]:


df.loc[(df["no_of_songs"] > 50) & (df["year_born"] >= 1942)]


# In[63]:


df.loc[(df["no_of_songs"] > 50) & (df["year_born"] >= 1942), ["Last Name", "Name"]]


# ## Manipulation von Spalten, Zeilen und bestimmten Einträgen

# **Hinzufügen einer Zeile zum Datensatz**

# In[64]:


from numpy import nan

df.loc[5] = ["Mickey", "Mouse", nan, 1928, nan]
df


# In[65]:


df.dtypes


# Beachten Sie, dass sich die Variable `dead` geändert hat. Ihre Werte änderten sich von `True`/`False` zu `1.0`/`0.0`. Folglich änderte sich ihr `dtype` von `bool` zu `float64`.

# **Hinzufügen einer Spalte zum Datensatz**

# In[66]:


from datetime import datetime

datetime.today()


# In[67]:


now = datetime.today().year
now


# In[68]:


df["age"] = now - df.year_born
df


# **Einen bestimmten Eintrag ändern**

# In[69]:


df.loc[5, "Name"] = "Minnie"


# In[70]:


df


# ## Plotten

# Die Plotting-Funktionalität in Pandas basiert auf Matplotlib. Es ist recht praktisch, den Visualisierungsprozess mit der grundlegenden Pandas-Darstellung zu beginnen und zu matplotlib zu wechseln, um die Pandas-Visualisierung anzupassen.

# ### `plot`-Methoden

# In[71]:


# dieser Aufruf bewirkt, dass die Zahlen unter den Codezellen eingezeichnet werden
get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


df


# In[73]:


df[["no_of_songs", "age"]].plot()


# In[74]:


df["dead"].plot.hist()


# In[75]:


df["age"].plot.bar()


# > __...einige Anmerkungen zum Plotten mit Python__

# Das Plotten ist ein wesentlicher Bestandteil der Datenanalyse. Die Welt der Python-Visualisierung kann jedoch ein frustrierender Ort sein. Es gibt viele verschiedene Optionen, und die Auswahl der richtigen ist eine Herausforderung. (Wenn Sie sich trauen, werfen Sie einen Blick auf die <a href="https://github.com/rougier/python-visualization-landscape">Python-Visualisierungslandschaft</a>).
# 
# <a href="https://matplotlib.org/">matplotlib</a> ist wahrscheinlich die bekannteste Python-Bibliothek für 2D-Diagramme. Mit ihr lassen sich plattformübergreifend Zahlen in Publikationsqualität in einer Vielzahl von Formaten und interaktiven Umgebungen erstellen. Allerdings ist matplotlib aufgrund der komplexen Syntax und der Existenz zweier Schnittstellen, einer **MATLAB-ähnlichen zustandsbasierten Schnittstelle** und einer **objektorientierten Schnittstelle**, schwer zugänglich. Daher gibt **es immer mehr als eine Möglichkeit, eine Visualisierung zu erstellen**. Eine weitere Quelle der Verwirrung ist die Tatsache, dass matplotlib gut in andere Python-Bibliotheken integriert ist, wie z. B. <a href="https://pandas.pydata.org/index.html">pandas</a>, <a href="http://seaborn.pydata.org/index.html">seaborn</a>, <a href="https://xarray.pydata.org/en/stable/">xarray</a> und andere. Daher gibt es Verwirrung darüber, wann man die reine matplotlib oder ein Tool, das auf matplotlib aufbaut, verwenden sollte.
# 
# Wir importieren die `matplotlib`-Bibliothek und das `pyplot`-Modul von matplotlib mit den folgenden kanonischen Befehlen
# 
# `import matplotlib as mpl`
# `import matplotlib.pyplot as plt`
# 
# In Bezug auf die Terminologie von matplotlib ist es wichtig zu verstehen, dass die `Figure` das endgültige Bild ist, das eine oder mehrere `Axes` enthalten kann, und dass die  `Axes` eine individuelle Darstellung repräsentieren.
# 
# Um ein `Figure`-Objekt zu erstellen, rufen wir
# 
# `fig = plt.figure()` auf.
# 
# Ein bequemerer Weg, ein `Figure`-Objekt und ein `Axes`-Objekt auf einmal zu erstellen, ist jedoch der Aufruf
# 
# `fig, ax = plt.subplots()` 
# 
# Dann können wir das `Axes`-Objekt verwenden, um Daten für die Darstellung hinzuzufügen.

# In[76]:


import matplotlib.pyplot as plt

# Erzeuge Figure und Axes Objekt
fig, ax = plt.subplots(figsize=(10, 5))

# plot die Daten und referenzier das Axes Objekt
df["age"].plot.bar(ax=ax)

# Passe das Axes Objekt an
ax.set_xticklabels(df["Name"], rotation=0)
ax.set_xlabel("")
ax.set_ylabel("Age", size=14)
ax.set_title("The Beatles and ... something else", size=18)


# Beachten Sie, dass wir nur an der Oberfläche der Plot-Möglichkeiten mit Pandas kratzen. In der Online-Dokumentation von Pandas (<a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html">hier</a>) finden Sie einen umfassenden Überblick.
