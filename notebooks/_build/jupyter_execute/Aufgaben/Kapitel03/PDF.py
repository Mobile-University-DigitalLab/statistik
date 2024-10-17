#!/usr/bin/env python
# coding: utf-8

# # Wahrscheinlichkeitsdichtefunktion

# 1. Nennen Sie die $3$ Haupteigenschaften von Wahrscheinlichkeitsdichtfunktionen (PDF).
# 2. Wie kann das Flächenintegral über eine Wahrscheinlichkeitsdichtfunktion interpretiert werden?
# 3. Mit welcher Python Funktion kann die Warscheinlichkeitsdichtefunkten der Normaverteilung bestimmt werden?

# -------------------------------------------------------

# ## Lösungen

# ```{toggle}
# * Eine PDF wird immer auf oder über der horizontalen Achse gezeichnet
# * Die Gesamtfläche zwischen einer PDF und der horizontalen Achse ist gleich $1$ und somit liegt jeder Wert in jedem Teilintervall der PDF im Bereich von $0$ bis $1$
# * Alle möglichen Beobachtungen der Variablen, die innerhalb eines bestimmten Bereichs liegen, entsprechen der entsprechenden Fläche unter der Dichtefunktion und können als prozentueller Anteil ausgedrückt werden. 
# 
# ({cite:p}`Papula2011` s.327)
# ```

# ```{toggle}
# Das Integral über einer Wahrscheinlichkeitsdichte entspricht der Wahrscheinlichkeit das die entsprechende Zufallsvariable in diesem Bereich liegt. 
# ```

# ```{toggle}
# Das Paket Scipy liefert die für alle geläufigen Wahrscheinlichkeitsdichtefunktionen, so auch für die Normalverteilung die Funktion `pdf` (z.B. `scipy.stats.norm.pdf()`, siehe auch https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
# ```
