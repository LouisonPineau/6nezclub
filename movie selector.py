# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:54:24 2026

@author: pineaulo
"""

from io import BytesIO

import requests
import pandas as pd

#Request to get the data from the google sheet
r = requests.get('https://docs.google.com/spreadsheets/d/e/2PACX-1vQ-I3jNSGm4lQ4woB_RduhnQ-Y_23h-okao2BBRxGe4ESRfOatDNtl4ibZcN8HETMPDmvN_FuCzeKrC/pub?gid=0&single=true&output=csv')
data = r.content

#Format the data into a panda dataframe
df = pd.read_csv(BytesIO(data), index_col=0)

#Resampling to only take unseen movies (google sheet will need to be updated regularly).
#For this, we create a new dataframe containing only the rows for which
#statut = Pas encore vu
df1 = df[df['Statut'] == 'Pas encore vu']

#Then, using groupby, we sort by Catégorie and sample one item from each category.
#For this, we use apply(lambda x: x.function()) to apply a simple function (here sample())
#Each sampled item is appended to a dictionnary.
df2 = df1.groupby('Catégorie').apply(lambda grp: grp.sample(n=1))

#We format df2 into a standard panda dataframe.
df2 = df2.reset_index(drop=True)

#Finally, we print the name of the slected movies.
print('Les films sélectionnés sont', df2.get('Titre'))


    


