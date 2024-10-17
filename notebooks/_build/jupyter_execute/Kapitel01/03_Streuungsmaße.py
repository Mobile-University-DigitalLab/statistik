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


# # Streuungsmaße

# In[2]:


import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde


# Die Maße der zentralen Tendenz, wie Mittelwert, Median und Modus, geben nicht das ganze Bild der Verteilung eines Datensatzes wieder. Zwei Datensätze mit identischem Mittelwert können völlig unterschiedliche Streuungen aufweisen. Die Streuung der Beobachtungswerte des einen Datensatzes kann viel größer oder kleiner sein als die des anderen Datensatzes. Daher ist der Mittelwert, Median oder Modus allein in der Regel kein ausreichendes Maß, um die Form der Verteilung eines Datensatzes aufzuzeigen. Wir benötigen auch ein Maß, das Informationen über die Variation zwischen den Datenwerten liefert. Diese Maße werden als **Streuungsmaße** bezeichnet. Die Maße der zentralen Tendenz und der Streuung ergeben zusammengenommen ein besseres Bild eines Datensatzes als die Maße der zentralen Tendenz allein ({cite:p}`fahrmeirstatistik` s.65).

# ## Varianz und Standardabweichung

# Die **Varianz** ist die Summe der quadrierten Abweichungen vom Mittelwert. Die Varianz für Populationsdaten wird mit $\sigma^2$ bezeichnet (gelesen als Sigma-Quadrat), und die für Stichprobendaten berechnete Varianz wird mit $s^2$ bezeichnet. 
# 
# $$ \sigma^2 = \frac{\sum_{i=1}^n (x_i - \mu)^2}{N} $$
# 
# und
# $$ s^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1} $$
# 
# wobei $\sigma^2$ die Varianz der Grundgesamtheit und $s^2$ die Stichprobenvarianz ist. Die Größe $x_i-\mu$ oder $x_i-\bar{x}$ in den obigen Formeln wird als die Abweichung des $x_i$-Wertes $(x_1,x_2, \dots ,x_n)$ vom Mittelwert bezeichnet ({cite:p}`fahrmeirstatistik` s.64). 
# 
# Die <a href="https://de.wikipedia.org/wiki/Varianz_(Stochastik)">Standardabweichung</a> ist das gebräuchlichste Maß für die Streuung. Der Wert der Standardabweichung gibt an, wie eng die Werte eines Datensatzes um den Mittelwert herum gestreut sind. Im Allgemeinen zeigt ein niedriger Wert der Standardabweichung für einen Datensatz an, dass die Werte dieses Datensatzes über einen relativ kleineren Bereich um den Mittelwert herum verteilt sind. Im Gegensatz dazu zeigt ein größerer Wert der Standardabweichung für einen Datensatz an, dass die Werte dieses Datensatzes über einen relativ größeren Bereich um den Mittelwert herum gestreut sind ({cite:p}`fahrmeirstatistik` s.65).
# 

# In[3]:


import matplotlib.pyplot as plt

TITLESIZE = 17
SEED = np.random.seed(42)
SIZE = 100
x = np.linspace(0, 1, SIZE)

fig, ax = plt.subplots(ncols=2, figsize=(16, 6))

y = np.random.normal(loc=0.5, scale=1.0, size=SIZE)
ax[0].plot(x, y, "o")
ax[0].set_title("Geringe Abweichungen um den Mittelwert", size=TITLESIZE)

y = np.random.normal(loc=0.5, scale=3.0, size=SIZE)
ax[1].plot(x, y, "o")
ax[1].set_title("Grosse Abweichungen um den Mittelwert", size=TITLESIZE)

for _ax in ax:
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.axhline(0.5, color="k")
    _ax.set_ylim(-6, 6)
    _ax.legend(["Daten", "Mittelwert"])


# Die Standardabweichung erhält man durch Ziehen der Quadratwurzel aus der **Varianz**. Folglich wird die für Grundgesamtheitsdaten berechnete Standardabweichung mit $\sigma$ und die für Stichprobendaten berechnete Standardabweichung mit $s$ bezeichnet. 
# 
# $$ \sigma = \sqrt{\frac{\sum_{i=1}^n (x_i - \mu)^2}{N}} $$
# 
# und 
# 
# $$ s = \sqrt{\frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1}} $$ 
# 
# wobei $\sigma$ die Standardabweichung der Grundgesamtheit und $s$ die Standardabweichung der Stichprobe ist.
# 
# Als Übung berechnen wir für einige numerische Variablen, die im `students` Datensatz von Interesse sind, den Mittelwert, den Median, die Varianz und die Standardabweichung und stellen sie in einem Dataframe dar.

# In[4]:


# Lesen der Datei students.csv als Dataframe; nur die Spalten "age", "weight", "height" und "nc_score" Spalten werden eingelesen
students = pd.read_csv(
    "../../data/students.csv",
    index_col=0,
    usecols=["age", "height", "weight", "nc_score"],
)
# Zeige die ersten 10 Werte
students.head(10)


# Pandas verfügt über die Methode `agg`. Diese lässt uns sehr einfach verschiedene deskritptive Statisitken berechnen.

# In[5]:


students.agg(["mean", "median", "var", "std"])


# ### Verwendung der Standardabweichung
# Mit Hilfe des **Mittelwerts** und der **Standardabweichung** lässt sich der Anteil oder Prozentsatz der Gesamtbeobachtungen ermitteln, die in ein bestimmtes Intervall um den Mittelwert fallen.

# #### Tschebyscheff-Theorem
# Die <a href="https://de.wikipedia.org/wiki/Tschebyscheffsche_Ungleichung">Tschebyscheff Ungleichung</a> gibt eine untere Schranke für die Fläche unter einer Kurve zwischen zwei Punkten, die auf gegenüberliegenden Seiten des Mittelwerts und im gleichen Abstand vom Mittelwert liegen.

# > **Für jede Zahl $k$, die größer als $1$ ist, liegen mindestens $1-\frac{1}{k^2}$ der Datenwerte innerhalb von $k$ Standardabweichungen vom Mittelwert.**

# Lassen Sie uns Python verwenden, um ein Gefühl für den Tschebyscheff-Theorem zu bekommen.

# In[6]:


k = np.arange(1, 4.1, 0.1)
value = np.round((1 - (1 / k**2)) * 100)
chebyshev = pd.DataFrame({"k": k, "Prozent": value})
chebyshev


# Um es in Worte zu fassen: Für $k=2$ bedeutet das, dass mindestens **$75 \% $** der Datenwerte innerhalb von **2 Standardabweichungen** vom Mittelwert liegen.
# 
# Stellen wir das Tschebyscheff-Theorem mit Python dar: 

# In[7]:


fig, ax = plt.subplots()
chebyshev.plot.scatter(x="k", y="Prozent", ax=ax)
ax.set_title("Tschebyscheff-Theorem")
ax.grid()


# Das Theorem gilt sowohl für Stichproben- als auch für Grundgesamtheitsdaten. Die Tschebyscheffsche Ungleichung gilt für Verteilungen beliebiger Form. Sie kann jedoch nur für $k > 1$ verwendet werden. Denn wenn $k=1$ ist, ist der Wert von $1-\frac{1}{k^2}$ Null, und wenn $k < 1$ ist, ist der Wert von $1-\frac{1}{k^2}$ negativ ({cite:p}`fahrmeirstatistik` s.304). 

# #### Empirische Regel
# Während die Tschebyscheffsche Ungleichung auf jede Art von Verteilung anwendbar ist, gilt die **empirische Regel** nur für eine bestimmte Art von Verteilung, die so genannte **Gaußverteilung** oder **Normalverteilung**. Es gibt 3 Regeln: <br> <br> Bei einer Normalverteilung sind   
# 
# 1.  $68 \%$ der Beobachtungen innerhalb einer Standardabweichung des Mittelwerts.
# 2.  $95 \%$ der Beobachtungen innerhalb von zwei Standardabweichungen des Mittelwerts.
# 3.  $99,7 \%$ der Beobachtungen innerhalb von drei Standardabweichungen des Mittelwerts.
# 
# 

# Da wir inzwischen über genügend Hacking-Power verfügen, werden wir versuchen zu testen, ob die drei Regeln gültig sind. 
# 
# **(1) Erstens** werden wir die Funktion `random.normal()` in Python erforschen, um normalverteilte Daten zu erzeugen, und 
# 
# **(2) Zweitens** werden wir zu unserem `students` Datensatz zurückkehren um diese Regeln an diesem Datensatz zu validieren. 
# <br> <br> Die Normalverteilung gehört zur Familie der <a href="https://de.wikipedia.org/wiki/Wahrscheinlichkeitsma%C3%9F">stetigen Verteilungen</a>. In Python gibt es eine Vielzahl von Wahrscheinlichkeitsverteilungen <a href="https://docs.scipy.org/doc/scipy/reference/stats.html">(hier)</a>. Um Daten aus einer Normalverteilung zu erzeugen, kann man die Funktion `random.normal()` verwenden, die ein Zufallsvariablengenerator für die Normalverteilung ist.
# 
# Mit der Funktion `np.random.normal(loc=0.0, scale=1.0)` können wir Zufallsvariablen aus einer Normalverteilung mit einem gegebenen Mittelwert (Standard ist $0$) und einer Standardabweichung (Standard ist $1$) entnehmen. Mit dem Argument `size` können wir die Anzahl der erzeugten Zufallsvariablen bestimmen.

# In[8]:


np.random.normal(loc=0.0, scale=1.0, size=1)


# In[9]:


np.random.normal(loc=0.0, scale=1.0, size=1)


# In[10]:


np.random.normal(loc=0.0, scale=1.0, size=1)


# Wir können die Funktion ziemlich einfach bitten, hunderte oder tausende oder noch mehr (Pseudo-)Zufallszahlen zu ziehen:

# In[11]:


np.random.normal(loc=0.0, scale=1.0, size=10)


# In[12]:


np.random.normal(loc=0.0, scale=1.0, size=100)


# Wenn wir ein Histogramm dieser Zahlen erstellen, sehen wir die namensgebende glockenförmige Verteilung.

# In[13]:


y_norm = np.random.normal(loc=0.0, scale=1.0, size=100000)
bins = int(len(y_norm) / 1000)  # bestimmt die Anzahl der Bins
plt.hist(y_norm, bins=bins)
plt.ylabel("Absolute Häufigkeit")
plt.show()


# Wir kennen bereits den Mittelwert und die Standardabweichung der Werte in `y_norm`, da wir die Funktion `np.random.normal()` explizit mit `mean=0` und `sd=1` aufgerufen haben. Wir müssen also nur die Zahlen in `y_norm` zählen, die größer als $-1$ bzw. kleiner als $1$ und $2$ bzw. $-2$ und $3$ bzw. $-3$ sind, und sie zur Länge von `y_norm`, in unserem Fall $100.000$, in Beziehung setzen, um die drei oben genannten Regeln zu bestätigen.

# In[14]:


# Berechne Anzahl der Werte < 1 - Anzahl der Werte > -1 durch Gesamtanzahl
sd1 = sum((y_norm > -1) & (y_norm < 1)) / len(y_norm)

# Berechne Anzahl der Werte < 2 - Anzahl der Werte  > -2 durch Gesamtanzahl
sd2 = sum((y_norm > -2) & (y_norm < 2)) / len(y_norm)

# Berechne Anzahl der Werte < 3 - Anzahl der Werte > -3 durch Gesamtanzahl
sd3 = sum((y_norm > -3) & (y_norm < 3)) / len(y_norm)

print("sd1 :", sd1)
print("sd2 :", sd2)
print("sd3 :", sd3)


# Perfekte Übereinstimmung! Die drei empirischen Regeln sind offensichtlich gültig. Um unsere Ergebnisse zu veranschaulichen, stellen wir das Histogramm erneut dar und fügen einige Anmerkungen hinzu. Bitte beachten Sie, dass wir in der `hist()`-Funktion das Argument `density=True` setzen. Dies hat zur Folge, dass das resultierende Histogramm nicht mehr die Zählungen auf der y-Achse anzeigt, sondern die **Dichtewerte** (normalisierte Zählung geteilt durch Bin-Breite), was bedeutet, dass sich die Balkenbereiche zu $1$ summieren.

# In[15]:


fig, ax = plt.subplots()
ax.hist(y_norm, bins=bins)
ax.set_ylabel("Absolute Häufigkeit")
for e, std in enumerate([(-1, 1), (-2, 2), (-3, 3)]):
    plt.axvline(std[0], color=f"C{e+1}")
    plt.axvline(std[1], color=f"C{e+1}")


# Nun, lassen Sie uns an der **zweiten** Aufgabe arbeiten: Überprüfen Sie die drei empirischen Regeln anhand des `students` Datensatzes. Dazu müssen wir überprüfen, ob eine der numerischen Variablen im Studentendatensatz  normalverteilt ist. Wir beginnen mit der Extraktion numerischer Variablen von Interesse aus dem `students` Datensatz. Dann zeichnen wir Histogramme und beurteilen, ob die Variable normalverteilt ist oder nicht. Zunächst überprüfen wir jedoch den Datensatz, indem wir die Funktion `head()` aufrufen.

# In[16]:


# Lesen der Datei students.csv als Dataframe;
students = pd.read_csv(
    "../../data/students.csv",
    usecols=["age", "height", "weight", "score1", "score2", "salary"],
)
students.head(10)


# Um einen Überblick über die Form der Verteilung der einzelnen Variablen zu erhalten, verwenden wir die Methode `hist()`.

# In[17]:


cols = students.columns
fig, ax = plt.subplots(ncols=int(len(cols) / 2), nrows=2)
ax = np.ravel(ax)  # vereinfacht das iterieren durch die axes Objekte
titles = {
    "age": "Alter",
    "height": "Grösse",
    "weight": "Gewicht",
    "score1": "Punkte1",
    "score2": "Punkte2",
    "salary": "Gehalt",
}

for e, col in enumerate(cols):
    bins = int(students[col].nunique() / 2)
    ax[e].hist(students[col], bins=bins, density=True)
    ax[e].set_title(titles[col])
fig.tight_layout()


# Wir stellen sofort fest, dass einige Variablen positiv verzerrt sind, also schließen wir sie aus und behalten diejenigen, die normal verteilt zu sein scheinen.

# In[18]:


cols = ["height", "salary"]
fig, ax = plt.subplots(ncols=2)
ax = np.ravel(ax)  # vereinfacht das iterieren durch die axes Objekte
bins_fact = {"height": 2, "salary": 40}

for e, col in enumerate(cols):
    bins = int(students[col].nunique() / bins_fact[col])
    ax[e].hist(students[col], bins=bins)
    ax[e].set_title(titles[col])
fig.tight_layout()


# Nun, sowohl die Variable `height` als auch die Variable `salary` scheinen mehr oder weniger normalverteilt zu sein. Es ist also eine Frage des Geschmacks, welche Variable man für die weitere Analyse auswählt. Für den Moment bleiben wir bei der Gehaltsvariable und überprüfen, ob die drei oben genannten empirischen Regeln gültig sind. Wir wechseln zu Python und validieren diese Regeln, indem wir zunächst den Mittelwert und die Standardabweichungen berechnen. Bitte beachten Sie, dass die Gehaltsvariable Fehlwerte enthält, die mit `NA` gekennzeichnet sind. Daher schließen wir zunächst alle `NA`-Werte aus, indem wir die Funktion `dropna()` anwenden.

# In[19]:


salary = students["salary"].dropna()
salary


# In[20]:


print(f"Mittelwert des Gehalts:             {salary.mean()}")
print(f"1 Standardabweichung des Gehalts:   {salary.std()}")
print(f"2 Standardabweichungen des Gehalts: {2 * salary.std()}")
print(f"3 Standardabweichungen des Gehalts: {3 * salary.std()}")


# Wie in der obigen allgemeinen Beispielform zählen wir die Anzahl der Werte, die größer als $+1$ s.d. bzw. kleiner als $-1$ s.d. und $+2$ s.d. bzw. $-2$ s.d. und $+3$ s.d. bzw. $-3$ s.d. sind, und setzen sie in Beziehung zur Länge des Vektors, in unserem Fall $1753$.

# In[21]:


salary_mean = salary.mean()
salary_std = salary.std()

sd1 = (
    sum((salary > (salary_mean - salary_std)) & (salary < (salary_mean + salary_std)))
    / salary.shape[0]
)

sd2 = (
    sum(
        (salary > (salary_mean - 2 * salary_std))
        & (salary < (salary_mean + 2 * salary_std))
    )
    / salary.shape[0]
)


sd3 = (
    sum(
        (salary > (salary_mean - 3 * salary_std))
        & (salary < (salary_mean + 3 * salary_std))
    )
    / salary.shape[0]
)


print("sd1 :", sd1)
print("sd2 :", sd2)
print("sd3 :", sd3)


# Wow, ziemlich nah dran! Offensichtlich zeigt die Gehaltsvariable eine starke Tendenz zur Unterstützung der so genannten empirischen Regel. Wir stellen das Histogramm für die Variable `salary` dar, um unseren Eindruck zu bestätigen. 

# In[22]:


fig, ax = plt.subplots()
bins = int(salary.nunique() / 40)
ax.hist(salary, bins=bins)
ax.set_ylabel("Absolute Häufigkeit")
ax.set_xlabel("Gehalt")
for e in range(3):
    ax.axvline(salary_mean + (e + 1) * salary_std, color=f"C{e+1}")
    ax.axvline(salary_mean - (e + 1) * salary_std, color=f"C{e+1}")


# Wir können nun unseren Visualisierungsansatz erweitern, indem wir die **empirische Dichteschätzung** mit der Funktion `scipy_kernel.evaluate()` grafisch darstellen und ihre Form überprüfen. Wir stellen die empirische Dichteschätzung als gestrichelte Linie dar, indem wir das <a href="https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html">(Linientyp-Argument)</a> `'-.'` und eine Linienbreite von $3$ (Argument `linewidth=3.0`) setzen.

# In[23]:


fig, ax = plt.subplots()
bins = int(salary.nunique() / 40)
ax.hist(salary, bins=bins, density=True)
ax.set_ylabel("Dichte")
ax.set_xlabel("Gehalt")

# empirische Dichteschätzung
x_salary = np.linspace(salary.min(), salary.max(), 100)
scipy_kernel = gaussian_kde(salary)
dens_emp = scipy_kernel.evaluate(x_salary)
ax.plot(x_salary, dens_emp, color="k", linestyle="dashed", linewidth=3.0)

# Standardabweichungen
for e in range(3):
    ax.axvline(salary_mean + (e + 1) * salary_std, color=f"C{e+1}")
    ax.axvline(salary_mean - (e + 1) * salary_std, color=f"C{e+1}")


# Schließlich vergleichen wir unsere **empirische Dichteschätzung** mit der theoretischen **Wahrscheinlichkeitsdichtefunktion**, die auf dem tatsächlichen Mittelwert und der Standardabweichung der Daten `salary` basiert. Für einen besseren visuellen Vergleich wechseln wir zurück zu einer nicht eingefärbten Histogramm-Darstellung.

# In[24]:


fig, ax = plt.subplots()
bins = int(salary.nunique() / 40)
ax.hist(salary, bins=bins, density=True)
ax.set_ylabel("Dichte")
ax.set_xlabel("Gehalt")

# empirische Dichteschätzung
x_salary = np.linspace(salary.min(), salary.max(), 100)
scipy_kernel = gaussian_kde(salary)
dens_emp = scipy_kernel.evaluate(x_salary)
ax.plot(
    x_salary,
    dens_emp,
    color="k",
    linestyle="dashed",
    linewidth=3.0,
    label="Empirische Dichteschätzung",
)

# Wahrscheinlichkeitsdichtefunktion
pdf = norm.pdf(x_salary, loc=salary.mean(), scale=salary.std())
ax.plot(
    x_salary, pdf, color="k", linewidth=3.0, label="Wahrscheinlichkeitsdichtefunktion"
)

# Standardabweichungen
for e in range(3):
    ax.axvline(salary_mean + (e + 1) * salary_std, color=f"C{e+1}")
    ax.axvline(salary_mean - (e + 1) * salary_std, color=f"C{e+1}")
ax.legend()


# Wir können daraus schließen, dass `salary` im Datensatz der `students` ungefähr normalverteilt ist. Die Grafik zeigt jedoch, dass die Verteilung der Gehaltsvariablen leicht linksschief ist. Dies ist an der Abweichung zwischen der **empirischen Dichteschätzung** und der **Wahrscheinlichkeitsdichtefunktion** zu erkennen.

# ## Die Spannweite

# Die **Spannweite** als Maß für die Streuung ist einfach zu berechnen. Sie ergibt sich aus der Differenz zwischen dem größten und dem kleinsten Wert in einem Datensatz.
# 
# $$\text{Range} = \text{größter Wert} - \text{kleinster Wert}$$

# Betrachten wir unseren `students` Datensatz. Wir unterteilen den Datensatz so, dass er nur numerische Daten enthält.

# In[25]:


df = pd.read_csv("../../data/students.csv")
df.sample(10)


# Wir sind also an den Kategorien `age`, `height`, `weight` und `nc_score` interessiert. Wir verwenden die Methoden `min()` und `max()`, um das Minimum und Maximum der ausgewählten Variablen zu berechnen. Erneut greifen wir auf die `agg`-Methode zurück.

# In[26]:


summary = df[["age", "height", "weight", "nc_score"]].agg(["min", "max"])
summary


# Um nun die Spannweite für jede Variable zu berechnen, müssen wir nur eine Zeile von der anderen abziehen.

# In[27]:


summary.loc["Spannweite"] = summary.loc["max"] - summary.loc["min"]
summary


# Die Spannweite hat, wie der Mittelwert, den Nachteil, dass sie durch Ausreißer beeinflusst wird. Daher ist die Spannweite kein gutes Streuungsmaß für einen Datensatz, der Ausreißer enthält. Ein weiterer Nachteil der Verwendung der Spannweite als Streuungsmaß ist, dass ihre Berechnung nur auf zwei Werten basiert: Dem größten und dem kleinsten. Alle anderen Werte in einem Datensatz werden bei der Berechnung der Spanne ignoriert. Daher ist die Spannweite oftmals kein sehr zufriedenstellendes Maß für die Streuung ({cite:p}`fahrmeirstatistik` s.62).
