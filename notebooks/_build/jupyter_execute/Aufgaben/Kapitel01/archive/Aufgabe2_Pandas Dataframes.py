#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------
# -------------------------------------------------------
# ### Kapitel 1
# ### Aufgabenstellung 2 : Pandas Dataframes
# 
# -------------------------------------------------------
# -------------------------------------------------------

# 1. Nutzen Sie die im Lehrbeispiel besprochenen Methoden um den Dataframe im Kapitel Strukturierte Datensätze nachzubauen.
# <body>
# 
# <table border="1" width="500" height="200">
#   <tr>
#     <td width="300" height="100">id </td>
#     <td width="300" height="100">Name</td>
#     <td width="300" height="100">Age</td>
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>John</td>
#     <td>26</td>
#   </tr>
#   <tr>
#     <td>2</td>
#     <td>Alice</td>
#     <td>20</td>
#    </tr>
#   <tr>
#     <td>3</td>
#     <td>Mike</td>
#     <td>21</td>
#    <tr>
#     <td>4</td>
#     <td>Anne</td>
#     <td>25</td>
#   </tr>
# </table>
# 
# </body>
# </html>
# 
# 

# 2. Fügen Sie die Spalte `nc_score` $= [2.5 , 3, 2.2 , 1.0] $ zum ersten Dataframe `df` hinzu.

# 3. Wählen Sie alle Daten aus für die gilt `age` $\gt 23 $ und speichern Sie die Werte in den neuen Dataframe `df2`.

# -------------------------------------------------------

# ### Lösung :

# 1.

# In[1]:


import pandas as pd

# Erstelle Dataframe df
df = pd.DataFrame({'id' : [1,2,3,4], 'Name': ['John', 'Alice', 'Mike', 'Anne'], 'Age': [26, 20, 21, 25]})
df


# 2.

# In[2]:


df['nc_score'] = [2.5,3,2.2,1.0]
df


# 3.

# In[3]:


df2 = df.loc[df['Age'] < 23]
df2


# In[ ]:




