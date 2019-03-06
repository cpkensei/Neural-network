import numpy as np
import random
import pylab as pl
#Kohonen/SOM network unsupervised learning


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
#There will be 3 input neurons
#output layer will have 4 neurons

def initialize(num_out,dimention):
    return [[random.random() for k in range(dimention)] for i in range(num_out)]

def c_rate(num_epoch,current_c):
    return current_c - (1/num_epoch)

def winner(network,input_value):
    #input values are always in 3 dimention
    distance = 0
    dis_l = list()
    for k in range(len(network)):
        distance = 0
        for i in range(len(input_value)):
            distance += (input_value[i] - network[k][i])**2 #calculate distance
        dis_l.append(distance)
    return dis_l.index(min(dis_l))

def neighbor(network,input_value,winner):
    distance = 0
    dis_l = list()
    l = [i for i in range(len(network))]
    for k in range(len(network)):
        distance = 0
        for i in range(len(input_value)):
            distance += (input_value[i] - network[k][i])**2 #calculate distance
        dis_l.append(distance)
    for j in range(len(dis_l)):
        dis_l[j] -= dis_l[winner]
        dis_l[j] = abs(dis_l[j])
    d1 = dis_l.index(max(dis_l))
    d2 = winner
    l.remove(d1)
    l.remove(d2)
    return l
        
def train_network(dataset,learning_rate,num_epoch):
    network = initialize(4,3)
    c = learning_rate
    count = 0
    for i in range(len(dataset)):
        #update the weight value for winner node and its neighbor
        win_node = winner(network,dataset[i])
        n1 = neighbor(network,dataset[i],win_node)
        for j in range(len(dataset[i])):
            network[win_node][j] = network[win_node][j] + c*(dataset[i][j]-network[win_node][j])
            network[n1[0]][j] = network[n1[0]][j] + c*(dataset[i][j]-network[n1[0]][j])
            network[n1[1]][j] = network[n1[1]][j] + c*(dataset[i][j]-network[n1[1]][j])
        c = c_rate(num_epoch,c)
        count+=1
        if count != num_epoch:
            i = 0
        if count == num_epoch:
            return network

a = train_network(dataset,1.0,10)
