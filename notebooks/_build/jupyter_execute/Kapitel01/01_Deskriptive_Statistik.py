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

get_ipython().run_line_magic('run', '../../src/notebook_env.py')


# # Deskriptive Statistik

# In[2]:


import pandas as pd


# Angewandte Statistik lässt sich in zwei Bereiche unterteilen: <a href="https://de.wikipedia.org/wiki/Deskriptive_Statistik">deskriptive Statistik</a> und <a href="https://de.wikipedia.org/wiki/Statistische_Inferenz">Inferenzstatistik</a>. Die deskriptive Statistik umfasst Methoden zur Organisation, Darstellung und Beschreibung von Daten mit Hilfe von Tabellen, Diagrammen und Streuungsmaßen. Im Gegensatz dazu besteht die Inferenzstatistik aus Methoden, die Stichprobenergebnisse verwenden, um Entscheidungen oder Vorhersagen über eine Grundgesamtheit zu treffen ({cite:t}`Cramer2008` s.1–151, 231–312, {cite:t}`fahrmeirstatistik` s.10, 12 ).
# 
# Das Wort [univariat](https://de.wikipedia.org/wiki/Univariat) bezieht sich auf die Tatsache, dass nur eine Variable betrachtet wird. Der Hauptzweck der univariaten Statistik besteht darin, die Daten zu beschreiben und zusammenzufassen. Wenn zwei oder mehr Variablen analysiert werden, spricht man von [bivariater](https://de.wikipedia.org/wiki/Univariat#Verwendung_in_der_Mathematik) oder [multivariater Analyse](https://de.wikipedia.org/wiki/Multivariate_Verfahren) bzw. Statistik. In diesem Fall sind wir in erster Linie an den Beziehungen zwischen und unter einer Reihe von Variablen interessiert.
# 

# ## Strukturierte Datensätze

# ### Strukturierte Daten
# Bei <a href="https://de.wikipedia.org/wiki/Data_Science">Data Science</a> geht es um die Gewinnung von Wissen aus 
# <a href="https://de.wikipedia.org/wiki/Daten">Daten</a>. Daten sind eine spezifische Form von <a href="https://de.wikipedia.org/wiki/Information">Informationen</a> und weisen verschiedene Abstraktions- und Strukturniveaus auf (<a href="https://de.wikipedia.org/wiki/Datenmodell">strukturiert</a>, <a href="https://de.wikipedia.org/wiki/Semistrukturierte_Daten">halbstrukturiert</a> oder <a href="https://de.wikipedia.org/wiki/Unstrukturierte_Daten">unstrukturiert</a>). <br> Eine sehr verbreitete <a href="https://de.wikipedia.org/wiki/Datenstruktur">Datenstruktur</a> ist ein Array. In verschiedenen Bereichen gibt es andere Bezeichnungen für einen solchen Datentyp, die synonym verwendet werden, z. B. Matrix ({cite:t}`Lang2016`) in der Mathematik, <a href="https://de.wikipedia.org/wiki/Datenbanktabelle">Tabelle</a> in Datenbanken, <a href="https://de.wikipedia.org/wiki/Tabellenkalkulation">Tabellenkalkulation</a> und <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html">Dataframe</a> , der eine grundlegende Python-Objektklasse ist z. B. : (Pandas `DataFrame`). <br> Daten eines solchen Typs bestehen aus Beobachtungen und entsprechenden Variablen, die oft als Merkmale bezeichnet werden.

# |id|Name|Age|
# |---|---|---|
# |1|John|26|
# |2|Alice|20|
# |3|Mike|21|
# |4|Anne|25|

# In diesem Beispiel entsprechen die **Beobachtungen** (*Stichprobe* genannt) einer Anzahl von Personen. Jede beobachtete Person wird durch eine Reihe von **Variablen** (so genannte *Merkmale*) charakterisiert: Durch eine Identifikationsnummer (id), durch einen Namen und durch ein Alter. In unserem Beispiel ist es sehr einfach, sich durch einen Blick auf die Tabelle einen Gesamteindruck von den Daten selbst zu verschaffen. Wir erkennen sofort, dass es in unserer Stichprobe $4$ Personen gibt, zwei Frauen und zwei Männer. Außerdem sehen wir sofort, dass die jüngste Person $20$ Jahre alt ist und Alice heißt und die älteste Person $26$ Jahre alt ist und John heißt.
# 
# Anwendungen in der realen Welt enthalten jedoch oft eine große Menge an Daten. Hunderte, Tausende, Millionen oder sogar Milliarden von Beobachtungen, kombiniert mit Tausenden von Variablen, können einen Datensatz bilden. Für den Menschen ist es unmöglich, allein durch die Betrachtung solcher Datensätze irgendwelche Schlussfolgerungen über die Daten zu ziehen. Daher reduzieren wir die Daten auf eine überschaubare Größe, indem wir Tabellen erstellen, Diagramme zeichnen oder zusammenfassende Maße wie Durchschnittswerte berechnen. Diese Art von statistischen Methoden wird als **deskriptive Statistik** bezeichnet ({cite:p}`fahrmeirstatistik` s.10).

# ### Der `students` Datensatz
# In diesem Abschnitt werden wir einen Datensatz namens `students` untersuchen. Zunächst laden wir den Datensatz, geben ihm einen geeigneten Namen und verschaffen uns einen Eindruck von seiner Struktur und Größe, indem wir die Methode `info()` auf den Datensatz anwenden.

# In[3]:


# Lese Datei students.csv als Dataframe ein
df = pd.read_csv("../../data/students.csv")
df.info()


# Der Studentendatensatz besteht aus $8239$ Zeilen, von denen jede einen bestimmten Studenten repräsentiert, und $16$ Spalten, von denen jede einer Variable/einem Merkmal entspricht, das sich auf diesen bestimmten Studenten bezieht. Diese selbsterklärenden Variablen sind: *stud_id, name, gender, age, height, weight, religion, nc_score, semester, major, minor, score1, score2, online_tutorial, graduated, salary*. Neben dem jeweiligen Variablennamen listet die Methode `info()` die `Klasse` jeder einzelnen Variablen auf. Alle Objekte in Python haben eine Klasse, z. B. `numerische` Datentypen , die in die Unterklassen `(int)` Ganzzahlen , `(float)` Gleitkommazahlen und `(imag)` komplexe Zahlen, eingeteilt werden.
# 
# In den nächsten Abschnitten werden wir die deskriptiven Statistiken des `students` Datensatzes genauer untersuchen.
# 
