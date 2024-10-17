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


# # Würfelexperiment
# 
# 1. Generieren Sie $100.000$ (gleichwahrscheinliche) Würfe eines Würfels und berechnen Sie Mittelwert und Standardabweichung der Würfelsumme. 
# 
# 2. Wählen Sie aus den Würfelsummen $200$ Stichproben mit einer Stichprobengrösse von $50$ aus. Berechnen Sie den Standardfehler mit 
# 
# $$\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$$
# 
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;wobei $\sigma_{\bar{x}}$ als Standardfehler, $\sigma$ als Standardabweichung der Stichprobe und $\sqrt{n}$ als Wurzel aus der Stichprobengrösse 
# 
# 3. Wiederholen Sie das Experiment für $10$ Würfel

# __Hilfsfunktionen__

# In[2]:


import numpy as np


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


# Frage 2 ...


# In[5]:


# Frage 3 ...


# ## Lösungen

# In[6]:


# Experiment
N = 100000
experiment = dice_roll(N, seed=42)

# Statistik des Experiments
print(f"Mittelwert Würfelsumme: {np.mean(experiment)}")
print(f"Standardabweichung Würfelsumme: {np.std(experiment)}")


# In[7]:


# Stichprobenverteilung und Standardfehler
n = 50
sample_means = []
for i in range(200):
    sample = np.random.choice(experiment, n, replace=True)
    sample_means.append(np.mean(sample))
std_error = np.std(sample_means) / np.sqrt(n)
print(f"Der Standardfehler beträgt {np.round(std_error,4)}.")


# In[8]:


# experiment
N = 100000
experiment = []
for i in range(N):
    roll = dice_roll(nrolls=10)
    roll_sum = np.sum(roll)
    experiment.append(roll_sum)

# Statistik des Experiments
print("Mittelwert Würfelsumme:", np.mean(experiment))
print("Standardabweichung Würfelsumme:", np.std(experiment))

# Stichprobenverteilung und Standardfehler
n = 50
sample_means = []
for i in range(200):
    sample = np.random.choice(experiment, n, replace=True)
    sample_means.append(np.mean(sample))
std_error = np.std(sample_means) / np.sqrt(n)
print(f"Der Standardfehler beträgt {np.round(std_error,4)}.")

