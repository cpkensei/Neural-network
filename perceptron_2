import random
import A1452 as a
import math
#read the data from testSeeds.csv file
def read_test_data():
    dataset1 = []
    f = open("testSeeds.csv","r")
    for lines in f:
        dataset1.append(lines)

    for i in range(len(dataset1)):
        dataset1[i] = dataset1[i].split(',')
        dataset1[i][-1] = dataset1[i][-1][0]
    return dataset1

test_dataset = read_test_data()

#the accuracy of neurons
#out_DE is the accuracy that only based on length and width

def out_DE(weight_low,weight_high,attr1,attr2,dataset):
    l = []
    k_of_func = float(weight_low[1])/-(float(weight_low[2]))
    b_of_func = float(weight_low[0])/-(float(weight_low[2]))    
    k2 = float(weight_high[1])/-(float(weight_high[2]))
    b2 = float(weight_high[0])/-(float(weight_high[2]))
    for i in range(len(dataset)):
        if float(dataset[i][attr1]) * k_of_func + b_of_func > float(dataset[i][attr2]):   ##class3
            l.append("3" == (dataset[i][-1]))
        else:
            if(float(dataset[i][attr1]) * k_of_func + b_of_func <= float(dataset[i][attr2])) and (float(dataset[i][attr1]) * k2 + b2 >= float(dataset[i][attr2])) :
                l.append("1"  == str(dataset[i][-1]))                                     ##class1
            else:
                if (float(dataset[i][attr1]) * k_of_func + b_of_func <= float(dataset[i][attr2])) and (float(dataset[i][attr1]) * k2 + b2 <= float(dataset[i][attr2])):
                    l.append("2" == str(dataset[i][-1]))
    return (l.count(True)/(l.count(True)+l.count(False)))

#uncomment this line to see the accuracy of this single neuron
#print(out_DE(a.v1,a.b1,3,4,test_dataset))

#out_AB is the accuracy that only based on area and perimeter 
def out_AB(weight_low,weight_high,attr1,attr2,dataset): ##takes an vertor
    l = []
    k_of_func = float(weight_low[1])/-(float(weight_low[2]))
    b_of_func = float(weight_low[0])/-(float(weight_low[2]))
    k2 = float(weight_high[1])/-(float(weight_high[2]))
    b2 = float(weight_high[0])/-(float(weight_high[2]))
    for i in range(len(dataset)):
        if float(dataset[i][attr1]) * k_of_func + b_of_func > float(dataset[i][attr2]):   ##class3
            l.append("3" == (dataset[i][-1]))
        else:
            if(float(dataset[i][attr1]) * k_of_func + b_of_func <= float(dataset[i][attr2])) and (float(dataset[i][attr1]) * k2 + b2 >= float(dataset[i][attr2])) :
                l.append("1"  == str(dataset[i][-1]))                                  ##class1

            else:
                if(float(dataset[i][attr1]) * k_of_func + b_of_func <= float(dataset[i][attr2])) and (float(dataset[i][attr1]) * k2 + b2 <= float(dataset[i][attr2])):
                    l.append("2" == str(dataset[i][-1]))
    return (l.count(True)/(l.count(True)+l.count(False)))

#uncomment this line to see the accuracy of this single neuron
#print(out_AB(a.v2,a.b2,0,1,test_dataset))

#out_FG is the accuracy that only based on the last 2 attributes of wheats
def out_FG(weight_low,weight_high,attr1,attr2,dataset): ##takes an vertor
    l = []
    k_of_func = float(weight_low[1])/-(float(weight_low[2]))  #yellow line - identifying class 2
    b_of_func = float(weight_low[0])/-(float(weight_low[2]))
    k2 = float(weight_high[1])/-(float(weight_high[2]))
    b2 = float(weight_high[0])/-(float(weight_high[2]))
    for i in range(len(dataset)):
        if float(dataset[i][attr1]) * k_of_func + b_of_func < float(dataset[i][attr2]):   ##class2
            l.append("2" == (dataset[i][-1]))
        else:
            if (float(dataset[i][attr2]) - b2) / (k2) > float(dataset[i][attr1]):
                l.append("1" == dataset[i][-1])
            else:
                l.append("3" == dataset[i][-1])
        
    return (l.count(True)/(l.count(True)+l.count(False)))

#uncomment this line to see the accuracy of this single neuron
#print(out_FG(a.b3,a.v3,5,6,test_dataset))

#only based on the 3rd attribute
def out_C(dataset):
    l = []
    for i in range(len(dataset)):
        if float(dataset[i][2]) >0.8529 and float(dataset[i][2]) <0.9183:
            l.append("1" == dataset[i][-1])
        else:
            if float(dataset[i][2]) >0.8452 and float(dataset[i][2]) <0.9081:
                l.append("2" == dataset[i][-1])
            else:
                l.append("3" == dataset[i][-1])
    return (l.count(True)/(l.count(True)+l.count(False)))

#uncomment this line to see the accuracy of this single neuron
#print(out_C(test_dataset))

#This function is used to test a single wheat's class
#takes all 7 attributes of wheat
#return its class
def guess_class(lst):
    l = []
    k_of_func = float(a.v1[1])/-(float(a.v1[2]))
    b_of_func = float(a.v1[0])/-(float(a.v1[2]))    
    k2 = float(a.b1[1])/-(float(a.b1[2]))
    b2 = float(a.b1[0])/-(float(a.b1[2]))
    if float(lst[3]) * k_of_func + b_of_func > float(lst[4]):   ##class3
        l.append(3)
    else:
        if(float(lst[3]) * k_of_func + b_of_func <= float(lst[3])) and (float(lst[3]) * k2 + b2 >= float(lst[4])) :
            l.append(1)                                     ##class1
        else:
            if (float(lst[3]) * k_of_func + b_of_func <= float(lst[4])) and (float(lst[3]) * k2 + b2 <= float(lst[4])):
                l.append(2)
    k_of_func = float(a.v2[1])/-(float(a.v2[2]))
    b_of_func = float(a.v2[0])/-(float(a.v2[2]))    
    k2 = float(a.b2[1])/-(float(a.b2[2]))
    b2 = float(a.b2[0])/-(float(a.b2[2]))
    if float(lst[0]) * k_of_func + b_of_func > float(lst[1]):   ##class3
            l.append(3)
    else:
            if(float(lst[0]) * k_of_func + b_of_func <= float(lst[1])) and (float(lst[0]) * k2 + b2 >= float(lst[1])) :
                l.append(1)                                  ##class1

            else:
                if(float(lst[0]) * k_of_func + b_of_func <= float(lst[1])) and (float(lst[0]) * k2 + b2 <= float(lst[1])):
                    l.append(2)
    k_of_func = float(a.v3[1])/-(float(a.v3[2]))  #yellow line - identifying class 2
    b_of_func = float(a.v3[0])/-(float(a.v3[2]))
    k2 = float(a.b3[1])/-(float(a.b3[2]))
    b2 = float(a.b3[0])/-(float(a.b3[2]))
    if float(lst[5]) * k_of_func + b_of_func < float(lst[6]):   ##class2
            l.append(2)
    else:
            if (float(lst[6]) - b2) / (k2) > float(lst[5]):
                l.append(1)
            else:
                l.append(3)

    return max(set(l), key =l.count)
    
##test its final accuracy (it has a range, each time you execute it, you get a
##different answer
'''
uncomment these following lines to test the accuracy
'''
l = []
for i in range(len(test_dataset)):
    l.append(str(guess_class(test_dataset[i])) == str(test_dataset[i][-1]))
print("\n" + "Final accuracy is: ")
print((l.count(True)/(l.count(True)+l.count(False))))

#final accuracy: 0.69 to 0.85

#write to output file
#please clear the output file if you want to run the following codes again
#uncomment to re-write the outputs of my neural network

##f = open("output.txt","w")
##for x1 in range(len(test_dataset)):
##    f.write("I guess: " + str(guess_class(test_dataset[x1])) + "\t" + "The real class is: " + str(test_dataset[x1][-1]) + "\n")
##f.close()

    
    


