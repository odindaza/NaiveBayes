# -*- coding: utf-8 -*-
"""
Análisis de Datos

@author: odind
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
#Importamos los datos del archivo .csv
dataset = pd.read_csv('Cleaned-Data.csv')

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Conjunto de entrenamiento y conjunto de pruebas
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)
# Estandarización de escalas
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#Clasificación y entrenamiento
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)







