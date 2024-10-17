#!/usr/bin/env python
# coding: utf-8

# # Punkt- und Intervallschätzungen

# 1. Erklären Sie was man unter Punktschätzung beziehungsweise Intervallschätzung versteht.
# 2. Wie viel größer muß die Stichprobe bei einer Intervallschätzung sein um die Länge der Konfidenzintervalle zu halbieren?

# -------------------------------------------------------

# ### Lösung

# ```{toggle}
# **Punktschätzung**
# 
# Mit Schätzverfahren zur Punktschätzung wird versucht aus Stichproben möglichst genaue Näherungswerte für Grundgesamtheitsparameter zu finden. Man unterscheidet unspezifische Parameter, wie z.B. Lage- bzw. Streuungsparameter, Median, Quantille und Korrelation, und spezifische Parameter eines Verteilungsmodells, wie beispielsweise $\mu$ und $\sigma$ der Normalverteilung $N(\mu,\sigma)$ oder den Parameter $\lambda$ der Poissonverteilung $P(\lambda)$.
# 
# ({cite:p}`fahrmeirstatistik` s.338)
# 
# **Intervallschätzung**
# 
# Die Intervallschätzung gibt die Präzision des Schätzverfahrens für die entprechende Punktschätzung an. Es wird eine obere und untere Grenze um die Punktschätzung konstruiert in dem mit der Wahrscheinlichkeit $1-\alpha$ der Wert der Grundgesamtheit enthalten ist. $\alpha$ entspricht der *Irrtumswahrscheinlichkeit* und wird als Signifikanzniveau bezeichnet während $1-\alpha$ die *Überdeckungswahrscheinlichkeit* ist.
# 
# ({cite:p}`fahrmeirstatistik` s.356)
# ```

# ```{toggle}
# Die Breite des Konfidenzintervalls ist gegeben durch $CI: \text{Punktschätzung} \pm \text{kritischer Punkt} \times \frac{s}{\sqrt{n}}$. Der Stichprobenumfang geht also als Faktor $\frac{1}{\sqrt{n}}$ in die Gleichung ein. Um die Breite zu halbieren (bei ungefähr gleicher Stichprobenstandardabweichung) muß $n$ also vervierfacht werden. $\frac{1}{\sqrt{4n}} = \frac{1}{2\sqrt{n}}$
# ```
