#!/usr/bin/env python
# coding: utf-8

# # Lineare Regression - Grundbegriffe

# 1. Was versteht man unter Residuen?
# 2. Was versteht man unter Homoskedastizität?
# 3. Was sind Hebelpunkte?

# -------------------------------------------------------

# ## Lösungen

# ```{toggle}
# Residuen sind die Abweichungen der beobachteten $y$-Werten zu den durch die berechnete Regressionsgerade vorhergesagten $y$-Werten und sind gegeben durch:
# 
# $$\hat \epsilon_i = y_i - \hat y_i \ , \ i = 1, \cdots, n$$
# 
# Sie ermöglichen es ein Maß für die Güte eines Regressionsmodells zu definieren.
# 
# ({cite:p}`fahrmeirstatistik` s.149)
# ```

# ```{toggle}
# Homoskedastizität ist dann gegeben, wenn die Varianz der Residuen ungefähr konstant ist. Alle Datenpunkte liegen in etwa in gleichem Abstand zur Regressionsgeraden.
# ```

# ```{toggle}
# Hebelpunkte sind einzelne Datenpunkt, die weit außerhalb der Datenmenge liegen. Durch ihr hohes Residuum, dass quadratische in die Methode der kleinsten Quadrate eingeht, haben sie einen überproportionale Effekt auf die resultierende Regressionskurve. 
# ```
