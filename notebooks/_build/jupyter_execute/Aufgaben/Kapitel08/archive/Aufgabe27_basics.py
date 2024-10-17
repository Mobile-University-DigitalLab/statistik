#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 8
# ### Aufgabenstellung 27 : Lineare Regression - Grundbegriffe
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Was versteht man unter Residuen?
# 2. Was versteht man unter Homoskedastizität?
# 3. Was sind Hebelpunkte?
# 4. Wozu dient Modelldiagnose und welche Möglichkeiten gibt es sie durchzuführen?

# -------------------------------------------------------

# ### Lösung

# **1.**
# 
# Residuen sind die Abweichungen der beobachteten $y$-Werten zu den durch die berechnete Regressionsgerade vorhergesagten $y$-Werten und sind gegeben durch:
# 
# $$\hat \epsilon_i = y_i - \hat y_i \ , \ i = 1, \cdots, n$$
# 
# Sie ermöglichen es ein Maß für die Güte eines Regressionsmodells zu definieren.
# 
# ({cite:p}`fahrmeirstatistik` s.149)

# **2.**
# 
# Homoskedastizität ist dann gegeben, wenn die Varianz der Residuen ungefähr konstant ist. Alle Datenpunkte liegen in etwa in gleichem Abstand zur Regressionsgeraden.

# **3.**
# 
# Hebelpunkte sind einzelne Datenpunkt, die weit außerhalb der Datenmenge liegen. Durch ihr hohes Residuum, dass quadratische in die Methode der kleinsten Quadrate eingeht, haben sie einen überproportionale Effekt auf die resultierende Regressionskurve. 

# **4.**
