import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
#load 2 files
t_dataset = pd.read_csv("trainSeeds.csv")
t_dataset1 = pd.read_csv("testSeeds.csv")

#t_labels(1) is thhe actual class for each wheat
t_axis = t_dataset.iloc[:, 0:7].values
t_labels = t_dataset.iloc[:, -1].values.reshape(t_dataset.iloc[:, -1].values.shape[0], 1)
t_axis1 = t_dataset1.iloc[:, 0:7].values
t_labels1 = t_dataset1.iloc[:, -1].values.reshape(t_dataset1.iloc[:, -1].values.shape[0], 1)


clsf = Perceptron(max_iter=100, tol=0.01, eta0 = 0.01)
clsf = clsf.fit(t_axis, t_labels.squeeze())
#m_predic is what the class that the tool guesses to be
m_predic = clsf.predict(t_axis1)
#generates accuracy and confusion matrix
c = confusion_matrix(t_labels1.squeeze(), m_predic)
a = accuracy_score(t_labels1.squeeze(), m_predic)

print(c, a)
