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


# # Die kontinuierliche gleichmäßige Verteilung

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform


# Die <a href="https://de.wikipedia.org/wiki/Stetige_Gleichverteilung">Gleichverteilung</a> ist die einfachste Wahrscheinlichkeitsverteilung, aber sie spielt eine wichtige Rolle in der Statistik, da sie bei der Modellierung von Zufallsvariablen sehr nützlich ist. Die Gleichverteilung ist eine kontinuierliche Wahrscheinlichkeitsverteilung und befasst sich mit Ereignissen, deren Auftreten gleich wahrscheinlich ist. Die kontinuierliche Zufallsvariable $X$ gilt als gleichmäßig verteilt oder hat eine rechteckige Verteilung auf dem Intervall $[a \ $,$ \ b]$. Wir schreiben $X \sim U(a \ $,$ \ b)$, wenn seine Wahrscheinlichkeitsdichtefunktion gleich $f(x)=\frac{1}{b-a},x \in [a \ $,$ \ b]$ und ansonsten gleich $0$ ist ({cite:t}`Papula2011` s.331).

# $$f(x) =
# \begin{cases}
# \frac{1}{b-a},  & \text{wenn   $a \le x \le b$} \\[2ex]
# 0, & \text{wenn   $x < a$   oder   $x > b$}
# \end{cases}
# $$

# Die folgende Abbildung zeigt eine kontinuierliche Gleichverteilung $X \sim U(-2 \ $,$ \ 0,8)$, also eine Verteilung, bei der alle Werte von $x$ innerhalb des Intervalls $[-2 \ $,$ \ 0,8] $ gleich $ \frac{1}{b-a}(=\frac{1}{0,8-(-2)}\approx 0,36)$ sind, während alle anderen Werte von $x$ gleich $0$ sind.

# In[3]:


x = np.linspace(-4.2, 4.2, num=1000)
fig, ax = plt.subplots()
ax.plot(x, uniform.pdf(x, -2, 2 + 0.8), linewidth=4)
ax.set_title(r"$X \sim U(-2, 0.8)$")
ax.set_ylabel("Wahrscheinlichkeitsdichte")


# Der Mittelwert und der Median sind gegeben durch 

# $$\mu = \frac{a+b}{2}\text{.}$$

# Die kumulative Dichtefunktion ist unten dargestellt und ergibt sich aus der Gleichung

# $$F(x) =
# \begin{cases}
# 0,  & \text{for $x < a$} \\[2ex]
# \frac{x-a}{b-a},  & \text{for $x \in [a,b)$} \\[2ex]
# 1, & \text{for $x \ge b$}
# \end{cases}
# $$

# In[4]:


x = np.linspace(-4.2, 4.2, num=1000)
fig, ax = plt.subplots()
ax.plot(x, uniform.cdf(x, -2, 2 + 0.8), linewidth=4)
ax.set_title(r"$X \sim U(-2, 0.8)$")
ax.set_ylabel("Kummulierte Wahrscheinlichkeitsdichte")


# ## Die kontinuierliche gleichmäßige Verteilung in Python

# Python ermöglicht den Zugriff auf die Gleichverteilung mit den Funktionen `uniform.pmf()`, `uniform.cdf()`, `uniform.ppf()` und `uniform.rvs()`. Wenden Sie die Funktion `dir()` auf diese Funktionen an, um weitere Informationen zu erhalten.
# 
# Die Funktion `uniform.rvs()` erzeugt Zufallsabweichungen der Gleichverteilung und wird als `uniform.rvs(loc, loc+scale, size)` geschrieben. Wir können auf einfache Weise $n$ Zufallsstichproben innerhalb eines beliebigen Intervalls erzeugen indem wir die Zahlenwerte für minimalen ($a$) und maximalen Wert ($b$) in `random.uniform(a,b, size)` einsetzen.

# In[5]:


u_rvs = np.random.uniform(-1, 1, size=40)
u_rvs


# Wir können die Dichtefunktion für $X \sim U(-2 \ $,$ \ 0,8)$ mit Hilfe der Funktion `uniform.rvs()` approximieren und die Ergebnisse als Histogramm darstellen.

# In[6]:


# Erzeuge gleichverteilte werte
u_rvs = np.random.uniform(-2, 0.8, size=10000)

# Plotte Histogramm
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_title("Histogramm für rand.uniform")
ax.set_ylabel("Wahrscheinlichkeitsdichte")
ax.hist(u_rvs, bins=20, edgecolor="k")


# Außerdem stellen wir sowohl das Dichtehistogramm von oben als auch die gleichmäßige Wahrscheinlichkeitsverteilung für das Intervall $[-2 \ $,$ \ 0,8]$ dar, indem wir die Funktion `uniform.pdf()` anwenden.

# In[7]:


# Erzeuge x-werte
x = np.linspace(-2.2, 1, num=1000)

# Erzeuge gleichverteilte werte
u_rvs = uniform.rvs(loc=-2, scale=2 + 0.8, size=10000)

# Plotte Histogramm und uniforme pdf
fig, ax = plt.subplots()
ax.set_title("Histogramm für rand.uniform")
ax.set_ylabel("Wahrscheinlichkeitsdichte")
ax.hist(u_rvs, bins=20, edgecolor="k", density=True)
ax.plot(x, uniform.pdf(x, -2, 2 + max(u_rvs)), linewidth=4, alpha=0.6)


# In[8]:


unif_mean = (-2 + 0.6) / 2
unif_mean


# Die Abbildung zeigt, dass unsere $10.000$ Stichproben, die nach dem Zufallsprinzip aus einer Gleichverteilung gezogen wurden (Histogramm), sich der Gleichverteilung $X \sim U(-2 \ $,$ \ 0,8)$ (Liniendiagramm).
# 
# Außerdem können wir die Funktion `uniform.cdf()` verwenden, um die Fläche unter der Kurve für einen bestimmten Schwellenwert zu berechnen, oder wir können die Funktion `uniform.ppf()` verwenden, um einen Schwellenwert für eine bestimmte Wahrscheinlichkeit zurückzugeben.

# ### Übung

# Betrachten wir die gleichmäßige Wahrscheinlichkeitsverteilung, die durch $X \sim U(-3 \ $,$ \ 5,5)$ gegeben ist.

# **Frage 1**

# Wie lautet der Mittelwert $\mu$ für die gegebene Gleichverteilung.

# In[9]:


unif_mean = (-3 + 5.5) / 2
unif_mean


# Der Mittelwert $\mu$ für die durch $X \sim U(-3 \ $,$ \ 5,5)$ gegebene gleichmäßige Wahrscheinlichkeitsverteilung beträgt $1,25$.

# **Frage 2**

# Welcher Wert von $x$ entspricht dem Wert, der die gegebene Gleichverteilung in zwei gleiche Teile teilt, oder anders ausgedrückt: $P(X \lt ?)=0,5$.

# In[10]:


p_50 = uniform.ppf(0.5, -3, 8.5)
p_50


# Das ist überhaupt keine Überraschung. Der Wert von $x$, der die Gleichverteilung in zwei gleiche Teile teilt, beträgt $1,25$ und ist somit gleich $\mu$.

# **Frage 3**

# Angenommen, die obige Verteilung beschreibt ein physikalisches Phänomen. Wie groß ist die Wahrscheinlichkeit, dass bei einer Messung des physikalischen Prozesses, der das Phänomen bestimmt, ein Wert $\ge 4$ gemessen wird, oder anders ausgedrückt: $P(X \ge 4)$. Aufgrund des Charakters einer Gleichverteilung ist die Messung eines beliebigen Wertes innerhalb des Intervalls $[-3 \ $,$ \ 5,5]$ gleich wahrscheinlich ist.
# 
# Wir werden diese Frage auf zwei Arten lösen, numerisch und analytisch. Um die Frage numerisch zu lösen, müssen wir zunächst ein Experiment durchführen. Wir wiederholen unsere Messung eine große Anzahl von Malen und zählen dann, wie oft wir einen Wert $\ge 4$ registriert haben
# . Dank der Leistungsfähigkeit von Python und dem integrierten Zufallszahlengenerator `uniform.rvs()` für gleichmäßig verteilte Daten) ist die Wiederholungsaufgabe sehr einfach, allerdings sollte man sich darüber im Klaren sein, dass in realen Anwendungen oft nur eine sehr begrenzte Anzahl von Messungen verfügbar ist.

# In[11]:


# Erzeuge gleichverteilte Zufallsvariablen und Variable count
u_rvs = uniform.rvs(-3, 8.5, size=10000)

# Zähle Werte größer gleich 4
count = sum(u_rvs >= 4)

# Dividiere durch Gesamtanzahl der Werte
count = count / len(u_rvs)
count


# Die Ergebnisse zeigen, dass etwa $18 \%$ der Messungen Werte $\ge 4$ ergeben.
# 
# Zweitens, um die Frage analytisch zu lösen, verwenden wir die kumulative Wahrscheinlichkeitsdichtefunktion, die in Python für gleichmäßige Verteilungen durch die Funktion `uniform.cdf()` implementiert ist. 
# Wir interessieren uns also für die Fläche unter der Kurve bis zum Wert von $x=4$.

# In[12]:


1 - uniform.cdf(4, -3, 8.5)


# Der analytische Ansatz ergibt ein Ergebnis von $0,1764706$ oder anders ausgedrückt, mit einer Wahrscheinlichkeit von $17,65 \%$ erhalten wir Werte $\ge 4$, also $P(X \ge 4) \approx 0,18$.
# 
# Es ist offensichtlich, dass beide Ansätze sehr ähnliche Ergebnisse liefern. Es ist jedoch zu beachten, dass das Ergebnis des numerischen Ansatzes eine Annäherung an das analytische Ergebnis darstellt. Bedenken Sie, dass die Qualität einer solchen Annäherung sehr stark von der Anzahl der Zufallsvariablen abhängt, aus denen die Stichprobe besteht, in unserem Fall von der Anzahl der Messungen.
