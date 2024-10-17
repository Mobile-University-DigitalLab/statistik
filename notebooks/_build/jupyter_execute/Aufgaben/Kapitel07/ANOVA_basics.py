#!/usr/bin/env python
# coding: utf-8

# # Einfaktorielle ANOVA Grundbegriffe

# Beschreiben Sie die Grundlagen der einfaktoriellen ANOVA.
# 
# 1. Wonach kann mittels ANOVA getestet werden? Nennen Sie mögliche praktische Anwendungen in der Wissenschaft.
# 2. Welche Bedingungen müssen für das Durchführen einer ANOVA gegeben sein?
# 3. Was sind die Einschränkungungen einer ANOVA?

# -------------------------------------------------------

# ## Lösungen

# ```{toggle}
# - Es kann einerseits darauf getestet werden ob drei oder mehr Gruppen sich in einem Grundgesamtheitsparameter unterscheiden. Man misst dieselbe abhängige Variable für unterschiedliche unabhängige Gruppen und überprüft, anhand des mittels $F$-Teststatistik errechneten $p$-Werts, ob sich mindestens eine dieser Gruppen statistisch voneinander unterscheiden.
# 
# - Mögliche Anwendungen für ANOVA sind Experimente mit Test- und Kontrollgruppen mit der Zielsetzung zu überprüfen ob ein Effekt in der Testgruppe eintritt. Dieses Versuchsdesign findet sich in vielen verschiedenen Bereichen wieder, vor allem aber in der Experimentalpsychologie, Wirtschaftswissenschaften und Medizin.
# ```

# ```{toggle}
# Folgende Bedingungen müssen für die ANOVA erfüllt sein:
# 
# 1. Die abhängige Variable (Zielgröße) muß metrisch skalieren.
# 2. Die Zielgröße der Grundgesamtheit sollte ungefähr normalverteilt sein.
# 3. Die Aufteilung in Gruppen durch mindestens eine unabhängige Variable (Faktor) muß möglich sein. 
# 4. Die Vergleichsgruppen müssen unabhängige Zufallsstichproben sein.
# 5. Die Vergleichsgruppen sollten in etwa gleiche Varianzen haben.
# 
# ({cite:p}`fahrmeirstatistik` s.485)
# ```

# ```{toggle}
# Die einfaktorielle ANOVA ist ein Omnibus-Verfahren. Daher kann der errechnete $p$-Wert nur aussagen ob sich eine oder mehrere der Gruppen statistisch signifikant von den anderen unterscheiden, aber nicht welche Gruppen.
# ```
