import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 1.area, 2.perimeter,
#3.compactness,
#4.length, 5.width
# 6.asymmetry coefficient,
#7.length of kernel groove,
#8.class 
data = pd.read_csv("./trainSeeds.csv", header=None)
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

def graph():
    plt.scatter(x[:55, 5], x[:55, 6], color='blue', marker='o', label='1')
    plt.scatter(x[55:110, 5], x[55:110, 6], color='red', marker='x', label='2')
    plt.scatter(x[110:, 5], x[110:, 6], color='green', marker='^', label='3')
    plt.plot([121/-133, 0], [0, 121/125.72],'r')
    plt.plot([153/-2.5, 0], [0, 153/28.36],'y')
    plt.xlabel('length')
    plt.ylabel('width')
    plt.legend(loc = 'upper left')
    plt.title('Raw Data')
    plt.show()

graph()
