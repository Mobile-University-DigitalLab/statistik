#!/usr/bin/env python
# coding: utf-8

# # Logistische Regression - Grundbegriffe

# 1. Was sind Odds, Log-Odds ($Logit$)? Auf welchen Wertebereich bilden sie ab?
# 
# 2. Vergleichen Sie lineare und einfache logistische Regression. Worin unterscheiden sie sich?
# 
# 3. Was ist die logistische Funktion? Wie hängt Sie mit der $logit$-Funktion zusammen?

# -------------------------------------------------------

# ## Lösungen

# ```{toggle}
# Odds ($o$) sind definiert als die Wahrscheinlichkeit, dass ein Ereignis eintritt, geteilt durch die Wahrscheinlichkeit, dass es nicht eintritt.
# 
# $$o = \frac{\pi}{1-\pi}$$
# 
# Mit anderen Worten, es ist ein Verhältnis von Erfolgen (oder Gewinnen) zu Verlusten (oder Misserfolgen). Die Odds bilden auf den Wertebereich $]0 \ $,$ \ +\infty[$ ab.
# 
# Log-odds sind definiert als der Logarithmus der Odds:
# 
# $$logit(\pi) =\log ( \frac{\pi}{1-\pi} )$$
# 
# Die Log-Odds bilden auf den Wertebereich $]-\infty , +\infty[$ ab.
# ```
# 

# ```{toggle}
# Im linearen Regressionsmodell wird versucht eine Gerade durch Datenpunkte zulegen, wobei der Abstand zu allen Punkten minimiert wird. Die $y$-Achse wird dabei als stetig angenommen. Das linearen Regressionsmodell kann wie folgt definiert werden:
# 
# $$y= \beta_0 + \beta_1x$$
# 
# $\beta_0$ und $\beta_1$ können über die Methode der kleinsten Quadrate geschätzt werden.
# 
# Das einfache logistische Modell wird wie folgt definiert:
# 
# $$\pi = \frac{1}{1+e^{-(\beta_0+ \beta_1x)}}$$
# 
# Es besitzt eine binäre $y$-Achse ($1$ oder $0$) und $\pi$ gibt die Wahrscheinlichkeit an für einen bestimmten $x$-Wert entweder $1$ oder $0$ zu erhalten. Um $\beta_0+ \beta_1x_1$ zu schätzen gehen wir analog zur linearen Regression vor, nur unterscheiden sie sich dadurch das wir in diesem Fall für die $logit$-Funktion:
# 
# $$logit(\pi) = \beta_0+ \beta_1x$$ 
# 
# $\beta_0$ und $\beta_1$ mittels Maximum-Likelihood Methode schätzen.
# ```
# 

# ```{toggle}
# Die logistische Funktion für einfache logistische Regression ist definiert als:
# 
# $$\pi = logit^{-1}(log \left( \frac{\pi}{1-\pi}\right)) = logit^{-1}(\eta) = \frac{e^{\eta}}{1+e^{\eta}} = \frac{1}{1+e^{-(\beta_0+ \beta_1x)}}$$
# 
# Die logistische Funktion ist die Inverse der $logit$-Funktion.
# ```
