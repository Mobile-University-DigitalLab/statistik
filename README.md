# Statistik und Machine Learning Modelle

Modul "Statistik und Machine Learning Modelle".

Ein umfassender Kurs über Statistik und Machine Learning Modelle.


# Installation
- bei github anmelden (siehe schnellanleitung.pdf)
- git installieren

- miniconda installieren:
- hier: `https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/` finden Sie die benötigten Installer. 
- Wählen Sie den **miniconda** Installer für Ihr Betriebssystem aus
- Wählen Sie den Installer für Python 3.12
- Downloaden Sie den Installer
- Führen Sie den Installer aus.
- Starten Sie die miniconda console (siehe schnellanleitung.pdf)

- Legen Sie ein Fork des Kurs-Repositories an (siehe schnellanleitung.pdf)
- Legen Sie einen Ordner an, in dem Sie arbeiten wollen, z.B. <Ihr-Home-Ordner>\mu_kurse
- "clonen" Sie das Repository (siehe schnellanleitung.pdf)

- cd <Ihr-Home-Ordner>\mu_kurse\statistik

# Python vorbereiten

Führen Sie folgenden Befehl auf der miniconda Konsole aus:
`conda env create -f environment.yml`

# Kurs ausführen

### Environment aktivieren
`conda activate statistik-env`

### Kurs als Jupyter Notebook starten
`jupyter lab`

# Kurs schließen

Geben sie in der miniconda Konsole ein: `conda deactivate`

# Contributing Guidelines

> [!IMPORTANT]  
> Die folgenden Informationen sind hauptsächlich für Entwickler des Kurses oder fortgeschrittene Nutzer gedacht, welche die Notebooks als Bundle zu einem Jupyter Book zusammenfügen möchten oder dieses als PDF generieren möchten.

Wenn Sie detaillierte Informationen zum Builden des kompletten Jupyter Books benötigen, sind diese hier hinterlegt:

[CONTRIBUTING](CONTRIBUTING.md)

