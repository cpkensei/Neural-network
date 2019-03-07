import numpy as np
import random
import pylab as pl
import math

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

def ec_distance(p1,p2):
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i])**2
    distance = math.sqrt(distance)
    return distance

#x range from [-1.29904731, 2.325876798]
#y range from [-0.878917218, 1.453850114]
#z range from [-1.006866904, 2.01089859]
def rand_centroids(num):
    l = []
    for i in range(num):
        a = random.randint(-1299,2325)/1000
        b = random.randint(-878,1453)/1000
        c = random.randint(-1006,2010)/1000
        l.append([a,b,c])
    return l


def train_network(num_centroids,dataset): #1st parameter is randomly selected initial centroids
                                          #2nd parameter is the dataset
    cluster1,cluster2 = [],[]
    sum1,sum2,sum3 = 0,0,0
    new_weight_1,new_weight_2 = [],[]
    E = 0
    old_E = 0
    while E <= old_E:
        old_E = E
        for i in range(len(dataset)):
            if ec_distance(num_centroids[0],dataset[i]) < ec_distance(num_centroids[1],dataset[i]):
                cluster1.append(dataset[i])
            else:
                cluster2.append(dataset[i])
        for i in range(len(cluster1)):
            sum1 += cluster1[i][0]
            sum2 += cluster1[i][1]
            sum3 += cluster1[i][2]
        new_weight_1 = [sum1/len(cluster1),sum2/len(cluster1),sum3/len(cluster1)]
        sum1,sum2,sum3 = 0,0,0
        for j in range(len(cluster2)):
            sum1 += cluster2[j][0]
            sum2 += cluster2[j][1]
            sum3 += cluster2[j][2]
        new_weight_2 = [sum1/len(cluster2),sum2/len(cluster2),sum3/len(cluster2)]

        for k in range(len(cluster1)):
            E += math.sqrt(ec_distance(cluster1[k],new_weight_1))
        for l in range(len(cluster2)):
            E += math.sqrt(ec_distance(cluster2[l],new_weight_2))
    return [new_weight_1,new_weight_2]

init = rand_centroids(2)
final_result = train_network(init,dataset)

        
    
    
    
    
        
        
        
    
