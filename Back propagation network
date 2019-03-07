import random
import math
import gc
from math import exp

def read_file(filename):
    ##Read from the file given
    dataset = []
    f = open(filename,"r")
    for lines in f:
        dataset.append(lines)

    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(',')
        dataset[i][-1] = dataset[i][-1][0:-1]
    del dataset[0]
    for k in dataset:
        del k[0]
    for j in dataset:
        for x in range(len(j)):
            j[x] = float(j[x])
    return dataset

dataset = read_file("GlassData.csv")

#A helper function that counts the categories of glass in given dataset
#For example: 20 instances of category 1, 15 instances of category 2....
def calculate(dataset):
    l = [0,0,0,0,0,0]
    for i in range(len(dataset)):
        if dataset[i][-1] == 1.0:
            l[0]+=1
        elif dataset[i][-1] == 2.0:
            l[1]+=1
        elif dataset[i][-1] == 3.0:
            l[2]+=1
        elif dataset[i][-1] == 5.0:
            l[3]+=1
        elif dataset[i][-1] == 6.0:
            l[4]+=1
        else:
            l[5]+=1
    return l

def generate_training(dataset):
    #assume training dataset is 70% of each category
    lst = calculate(dataset)
    count1,count2,count3,count5,count6,count7 = 0,0,0,0,0,0
    for i in range(len(lst)):
        lst[i] = int(lst[i] * (70/100))
    train_data = []
    for k in range(len(dataset)):
        if int(dataset[k][-1]) == 1 and count1 <= lst[0]:
            train_data.append(dataset[k])
            count1+=1
        if int(dataset[k][-1]) == 2 and count2 <= lst[1]:
            train_data.append(dataset[k])
            count2+=1
        if int(dataset[k][-1]) == 3 and count3 <= lst[2]:
            train_data.append(dataset[k])
            count3+=1
        if int(dataset[k][-1]) == 5 and count5 <= lst[3]:
            train_data.append(dataset[k])
            count5+=1
        if int(dataset[k][-1]) == 6 and count6 <= lst[4]:
            train_data.append(dataset[k])
            count6+=1
        if int(dataset[k][-1]) == 7 and count7 <= lst[5]:
            train_data.append(dataset[k])
            count7+=1
    return train_data

def generate(dataset):
    count1,count2,count3,count5,count6,count7 = 0,0,0,0,0,0
    #assume training dataset is 70% of each category,and 15% for validating and testing
    lst1 = calculate(dataset)
    lst2 = calculate(dataset)
    lst3,lst4,lst5 = [],[],[]
    for i in range(len(lst1)):
        lst1[i] = int(lst1[i] * (70/100)) #number of instances for training set
        lst2[i] = lst2[i] - lst1[i] #number of instances that can be used for validating and testing #15% of whole dataset can be used to be validation dataset
    for k in range(len(dataset)):
        if int(dataset[k][-1]) == 1 and k >= lst1[0] and count1 < lst2[0]:
            lst3.append(dataset[k])
            count1+=1
        if int(dataset[k][-1]) == 2 and k >= lst1[1] and count2 < lst2[1]:
            lst3.append(dataset[k])
            count2+=1
        if int(dataset[k][-1]) == 3 and k >= lst1[2] and count3 < lst2[2]:
            lst3.append(dataset[k])
            count3+=1
        if int(dataset[k][-1]) == 5 and k >= lst1[3] and count5 < lst2[3]:
            lst3.append(dataset[k])
            count5+=1
        if int(dataset[k][-1]) == 6 and k >= lst1[4] and count6 < lst2[4]:
            lst3.append(dataset[k])
            count6+=1
        if int(dataset[k][-1]) == 7 and k >= lst1[5] and count7 < lst2[5]:
            lst3.append(dataset[k])
            count7+=1
    #generate validating:
    for j in range(len(lst3)//2):
        lst4.append(lst3[j])
    #generate testing:
    for l in range(len(lst3)//2,len(lst3)):
        lst5.append(lst3[l])
    #the 1st return value is the validating dataset
    #the 2nd return value is the testing dataset
    return lst4,lst5

        
        
        
            
        

        
    
#make all instances and attributes have range from 0.0 to 1.0 to avoid calculation result becomes too large
def normalize(dataset):
    minmax = list()
    zip_data = zip(*dataset)
    for row in zip_data:
        minmax.append({'minimize': min(row), 'maximize': max(row)})
    #Start normalization
    for i in range(0, len(dataset)):
        for x in range(len(dataset[i]) - 1):  
            dataset[i][x] = (dataset[i][x] - minmax[x]['minimize']) / (minmax[x]['maximize'] - minmax[x]['minimize'])
    return dataset

new_data = normalize(dataset)

#calculate the activation of the node
def activate(weights, inputs):
	activation = weights[-1][0]
	for i in range(len(weights)-1):
		activation += weights[i][0] * inputs[i]
	return activation


def sigmoid(activation):
	ans1 = 1.0 / (1.0 + math.e**-activation)
	return ans1


def forward_propagate(network,row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'],inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
        if len(inputs)<9:#Avoid list out of range, append 0s to calculate activation
            a = 9 - len(inputs)
            for i in range(a):
                inputs.append(0)
    return inputs[0:6]
    
def sigmoid_derivative(output):
	return output * (1.0 - output)

#back propagate the output and error, then store in neurons
def backward_propagate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network)-1:#This is hidden/input layer
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j][0] * neuron['delta'])
                errors.append(error)
        else:#This is output layer
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
    return network

def update_weights(network, row, l_rate, momentum):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				deltaweight = l_rate * neuron['delta'] * inputs[j]
				if len(neuron['weights'][j]) < 2:
					neuron['weights'][j].append(deltaweight)
				else:
					neuron['weights'][j][1] = deltaweight
				neuron['weights'][j][0] += deltaweight + momentum * neuron['weights'][j][1]
			neuron['weights'][-1][0] += l_rate * neuron['delta']
	return network

def initialize_network(n_inputs, n_hidden, n_outputs):
	network = []
	#generates the basic architecture of network
	hidden_layer = [{'weights':[[random.random()] for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	#append hidden neurons to list
	output_layer = [{'weights':[[random.random()] for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	#each neuron has its random initial weight value
	return network

def predict(network, row):
    #predict the given instances' glass type
	outputs = forward_propagate(network, row)
	a = max(outputs)
	if outputs.index(a) < 3:
	    return outputs.index(a) + 1
	elif outputs.index(a) >= 3:
	    return outputs.index(a) + 2
            
            
#Training network takes about 0.5 - 3 minutes
#If the training is too slow, re-run this function, it should be faster
#To see whether the process is slow or fast, uncomment line245 to see the current accuracy
def train_network(network, train, l_rate, n_epoch, n_outputs, momentum,val_data):
    l = []
    accuracy = 0.0
    while accuracy < 0.6:
            for epoch in range(n_epoch):
                    vali_out = []
                    for row in train:
                            outputs = forward_propagate(network, row)
                            expected = [0 for i in range(n_outputs)]
##			1,2,3,5,6,7
                            if int(row[-1]) <= 3:
                                    expected[int(row[-1])-1] = 1   
                            expected[int(row[-1]-2)] = 1
                            backward_propagate(network, expected)
                            x1 = update_weights(network, row, l_rate, momentum)
                            network = x1
                    for i in range(len(val_data)):
                        l.append(predict(x1,val_data[i]) == int(val_data[i][-1]))
                    accuracy = l.count(True)/len(l)
##To see my training process, uncomment line245 which is the next line
##                    print(accuracy)
    return x1





train_data = generate_training(new_data)
val_data = generate(new_data)[0]
test_data = generate(new_data)[1]
#initializing network, with 9 inpuit neurons, 6 hidden neurons and 6 output neurons
x = initialize_network(9,6,6)
#train the network with training data and stop when validation data has accuracy higher than 60%
a = train_network(x,train_data,0.5,10,6,1,val_data)

def write_output_to_file(dataset,network):
    f = open("test_output.txt",'w')
    for i in range(len(dataset)):
        x = predict(a,dataset[i])
        actual = dataset[i][-1]
        f.write("Network predicts the type to be: "+str(x)+" and the actual type is; "+str(int(actual))+"\n")
    f.close()
    return

#write_output_to_file(test_data,a) #test the test_data
#write_output_to_file(new_data,a) #test all instances in dataset
#please remember to delete the output file "test_output.txt" after each time you test my code

def out_of_neurons(network):
    for layer in range(len(network)):
        if layer == 0:
            print("The hidden layer's output is \n")
        if layer == 1:
            print("The output layer's output is \n")
        for neuron in range(len(network[layer])):
            print("The output for No.",neuron+1, " neuron is: ",network[layer][neuron]['output'],"\n")

#uncomment line 283 (next line) too see each neuron's output
#out_of_neurons(a)
