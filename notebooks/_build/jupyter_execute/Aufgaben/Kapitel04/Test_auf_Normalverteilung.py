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

get_ipython().run_line_magic('run', '../../../src/notebook_env.py')


# # Test auf Normalverteilung
# Generieren Sie in 100.000 (gleichwahrscheinliche) Würfe eines Würfels und berechnen Sie Mittelwert und Standardabweichung der Würfelsumme. Wählen Sie aus den Würfelwürfen 200 Stichproben mit unterschiedlichen Stichprobenumfängen. Ab welchem Stichprobenumfang (`3, 5, 7, 10, 15, 20, 30, 50`) können wir dafon ausgehen, dass die Stichprobenverteilung des Mittelwertes normalverteilt ist. Nutzen sie zur Validierung der Hypothese den Wilk-Shapiro Test.

# **Hilfsfunktionen**

# In[2]:


import numpy as np
from scipy import stats


def test_for_normal_distribution(x, verbose=True):
    """Function to test if a sample is normally distributed.
    Therefore the Shapiro-Wilk test is employed. If the p-value is <0.05 we recject the null hypothesis and hence
    conclude that the data is not normally distrubuted for reference see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html"""
    shapiro_test = stats.shapiro(x)
    pvalue = shapiro_test.pvalue
    if verbose:
        print(f"p-value: {pvalue}")
        if pvalue < 0.05:
            print(
                f"The null hypothesis is rejected, the data is NOT normally distributed."
            )
        else:
            print(
                f"Given the data the null hypothesis cannot be rejected, the data is likely normally distributed."
            )
    return pvalue


def dice_roll(nrolls: int, nsides: int = 6, seed=None) -> list:
    """Function to simulate a dice roll
    params:
       nrolls: number of rolls/dices
       nsides: number of sides
    """
    if seed is not None:
        np.random.seed(seed)

    return [np.random.randint(1, nsides + 1) for x in range(nrolls)]


# -------------------------------------------------------

# In[3]:


# Frage 1 ...


# In[4]:


# experiment
N = 100000
seed = 42
experiment = dice_roll(N, seed=seed)

# Validierung
for n in [3, 5, 7, 10, 15, 20, 30, 50]:
    sample_means = []
    for i in range(200):
        sample = np.random.choice(experiment, n, replace=True)
        sample_means.append(np.mean(sample))
    print(f"\nSample size: {n}")
    pvalue = test_for_normal_distribution(sample_means)

