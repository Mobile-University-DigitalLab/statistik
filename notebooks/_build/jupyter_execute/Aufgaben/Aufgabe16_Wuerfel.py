#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Load the "autoreload" extension
get_ipython().run_line_magic('load_ext', 'autoreload')
# always reload modules
get_ipython().run_line_magic('autoreload', '2')
# black formatter for jupyter notebooks
#%load_ext nb_black
# black formatter for jupyter lab
get_ipython().run_line_magic('load_ext', 'lab_black')

get_ipython().run_line_magic('run', '../src/notebook_env.py')


# # Aufgabe 3
# 
# 1. Generieren Sie 100.000 (gleichwahrscheinliche) Würfe eines Würfels und berechnen Sie Mittelwert und Standardabweichung der Würfelsumme. Wählen Sie aus den Würfelsummen 200 Stichproben mit einer Stichprobengrösse von 50 aus. Berechnen Sie den Standardfehler mit 
# 
# $$\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$$
# 
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;wobei $\sigma_{\bar{x}}$ als Standardfehler, $\sigma$ als Standardabweichung der Stichprobe und $\sqrt{n}$ als Wurzel aus der Stichprobengrösse 
# 
# 2. Wiederholen Sie das Experiment für 10 Würfel

# **Importiere Module**

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# __Hilfsfunktionen__

# In[3]:


def dice_roll(nrolls: int, nsides: int = 6, seed=None) -> list:
    """Function to simulate a dice roll
    params:
       nrolls: number of rolls/dices
       nsides: number of sides
    """
    if seed is not None:
        np.random.seed(seed)

    return [np.random.randint(1, nsides) for x in range(nrolls)]


# ## Aufgabe 3.1
# 
# Generieren Sie in 100.000 (gleichwahrscheinliche) Würfe eines Würfels und berechnen Sie Mittelwert und Standardabweichung der Würfelsumme. Wählen Sie aus den Würfelwürfen 200 Stichproben vom Umfang 50 aus und berechnen den Standardfehler mit
# 
# $$\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$$
# 
# 
# wobei $\sigma_{\bar{x}}$ als Standardfehler, $\sigma$ als Standardabweichung der Stichprobe und $\sqrt{n}$ als Wurzel aus der Stichprobengrösse 

# In[4]:


## your code here ...


# In[5]:


# experiment
N = 100000
experiment = dice_roll(N, seed=42)

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


# ## Aufgabe 3.2
# Wiederholen Sie das Experiment für 10 Würfel

# In[6]:


## your code here ...


# In[7]:


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

