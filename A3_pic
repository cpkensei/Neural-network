import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def readfile():
    dataset = []
    f = open("./dataset_noclass.csv",'r')
    for lines in f:
        dataset.append(lines)
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(',')
        dataset[i][-1] = dataset[i][-1][0:-1]
    del dataset[0]
    for k in range(len(dataset)):
        for j in range(len(dataset[k])):
            dataset[k][j] = float(dataset[k][j])
    return dataset

dataset = readfile()
a = [[0.6777107384396686, 0.058637552796179125, 0.5403774601917521], [0.6777107384396686, 0.058637552796179125, 0.5403774601917521], [1.1171092162349936, -0.21402647882515724, -0.7203812512645174], [0.18754568590240012, 0.17203061387039986, 1.0913539166575998]]
def d3_points(dataset):
    x1 = dataset
    fig = plt.figure(figsize=(8, 8)) 
    ax = fig.gca(projection='3d') 
    for x in x1:
        ax.scatter(x[0],x[1],x[2],c='b')
    ax.scatter(a[2][0],a[2][1],a[2][2],c='r')
    ax.scatter(a[3][0],a[3][1],a[3][2],c='r')
    plt.savefig("d3_image.png")
    plt.show()


d3_points(dataset)
