#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 9
# ### Aufgabenstellung 31 : Logistische Funktion
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Zeigen Sie dass die logistische Funktion $logit^{-1}(\eta) = \frac{e^{\eta}}{1+e^{\eta}}$ die inverse Funktion zur $logit$-Funktion $\eta = logit(\pi) = log \left( \frac{\pi}{1-\pi}\right)$ ist.

# -------------------------------------------------------

# ### Lösung

# 1. logistische Funktion umwandeln

# $$\frac{e^{\eta}}{1+e^{\eta}} = \frac{e^{\eta}}{e^{\eta}}\frac{1}{e^{-\eta}+1} = \frac{1}{e^{-\eta}+1}$$

# 2. Logit Funktion in logistische Funktion einsetzen

# $$logit^{-1}(logit(\pi)) = logit^{-1}(log \left( \frac{\pi}{1-\pi}\right))) = $$

# $$ = \frac{1}{1+e^{-log \left( \frac{\pi}{1-\pi}\right)}} = \frac{1}{1+\frac{1}{\left( \frac{\pi}{1-\pi}\right)}} =$$

# $$ = \frac{1}{1+\frac{1 - \pi}{\pi}} = \frac{1}{\frac{\pi + 1 - \pi}{\pi}} = \pi$$

# 
