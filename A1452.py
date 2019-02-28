import random
import matplotlib.pyplot as plt


# A.area, B.perimeter, C.compactness, D.length, E.width
# F.asymmetry coefficient, G.length of kernel groove, H.class 
r = lambda x: round(x)

##Read the trainSeeds.csv file
dataset = []
f = open("trainSeeds.csv","r")
for lines in f:
    dataset.append(lines)

for i in range(len(dataset)):
    dataset[i] = dataset[i].split(',')
    dataset[i][-1] = dataset[i][-1][0]
    


def predictDE(data):
    if float(data[3]) < 5.6 and float(data[4]) < 3.2:
        return 3
    if float(data[4]) > 2.8 and float(data[4]) < 3.8 and float(data[3]) > 5.2 and float(data[3]) <6.0:
        return 1
    elif float(data[3]) > 6.0 and float(data[4]) > 3.6:
        return 2
    else:
        return 1

#This data is calculated by function IQR
##['2.04', '3.412'] F class 1  ##['2.858', '4.539'] F class 2
##['4.132', '5.462'] F class 3 ##['4.961', '5.299'] G class 1
##['5.894', '6.231'] G class 2 ##['5.046', '5.275'] G class 3
    
def predictFG(data):
    if float(data[5]) > 2.04 and float(data[5]) < 3.412 and float(data[6]) > 4.961 and float(data[6]) < 5.299:
        return 1
    elif(float(data[5]) > 2.858 and float(data[5]) < 4.539 and float(data[6]) > 5.894 and float(data[6]) < 6.231):
        return 2
    elif(float(data[5]) > 4.132 and float(data[5]) < 5.462 and float(data[6])> 5.046 and float(data[6]) < 5.275):
        return 3
    else:
        return random.randint(1,3)
    
#This data is calculayted by function IQR
##[13.94, 15.26] A class 1##[18.17, 19.31] A class 2
##[11.24, 12.21] A class 3##[14.06, 14.76] B class 1
##[16.05, 16.63] B class 2##[13, 13.47] B class 3 

def predictAB(dataset):
    if float(dataset[0]) > 13.94 and float(dataset[1]) >14.06 and float(dataset[0]) <15.26 and float(dataset[1]) < 14.76:
        return 1
    elif(float(dataset[0]) >18.17 and float(dataset[0])< 19.31 and float(dataset[1]) >16.05 and float(dataset[1])<16.63):
        return 2
    elif(float(dataset[0]) >11.24 and float(dataset[0])< 12.21 and float(dataset[1]) >13.00 and float(dataset[1])<13.47):
        return 3
    else:
        return random.randint(1,3)

#Use statistic method to approximately calculate the rannge of each class of wheat and use it to implement prediction function
def IQR(data):
    return [data[round(len(data)*0.25)], data[round(len(data)*0.75)]]
    

#w(t+1)= w(t) + learning_rate * (expected(t) - predicted(t)) * x(t)

#input layer (4 neurons) each neuron deals with 2 attributes (ex. height and width)
def train_weightsDE(init, c,in_data):
    w0 = init
    desired_output = float(in_data[-1])
    predict_output = predictDE(in_data)
    w1 = [None,None,None]
    w1[0] = float(w0[0]) + (24.5 * c *  (-1)*(abs(predict_output - desired_output)) * 1)                           #Calculate the weight of attribute d and e 
    w1[1] = abs(float(w0[1]) + (0.5*c *  (predict_output - desired_output) * float(in_data[3])))                   #which is length and width
    w1[2] = float(w0[2]) + (0.5*c *  (predict_output - desired_output) * float(in_data[4]))                       
    return w1

def train_weightsAB(init, c,in_data):
    w0 = init
    desired_output = float(in_data[-1])
    predict_output = predictAB(in_data)
    w1 = [None,None,None]                                                                                        #calculate the weight of area and perimeter (attribute a and b)
    w1[0] = float(w0[0]) + (30 * c *  (-1)*(abs(predict_output - desired_output)) * 2.18263)
    w1[1] = abs(float(w0[1]) + (0.5 * c *  (predict_output - desired_output) * float(in_data[0])))                 
    w1[2] = float(w0[2]) + (0.3 *c *  (predict_output - desired_output) * float(in_data[1]))
    return w1

def train_weightsFG(init,c,in_data):
    w0 = init
    desired_output = float(in_data[-1])           #calculate the weight of F.asymmetry coefficient, G.length of kernel groove
    predict_output = predictFG(in_data)
    w1 = [None,None,None]
    if(w0[1] or w0[2] == 0.0):
        w0[1] += 10
        w0[2] += 10
    w1[0] = float(w0[0]) + (30 * c *  (-1)*(predict_output - desired_output)) * 1
    w1[1] = ((float(w0[2]) * (-5.5)) - (float(w0[0])))/4
    w1[2] = ((-4) * float(w0[1]) - float(w0[0]))/5.5
    return w1

#This is a function tests the accuracy that linear function splits different class of wheats
#lst is the weight value, its length is 3 (w0,w1,and w2)
#dataset is the dataset we are using
#attr1 and attr2 are the attributes we are testing. For instance: length and width, then we type 3 and 4 which are index of attributes
#class1 and class2 are the classes we are tring to deal with
def test_weight(lst,dataset,attr1,attr2,class1,class2): #lst is [w0,w1,w2] dataset is a 2-dimentional list
    l_bottom = []
    l_top = []
    k_of_func = float(lst[1])/-(float(lst[2]))
    b_of_func = float(lst[0])/-(float(lst[2]))
    for i in range(len(dataset)):
        if float(dataset[i][attr1]) * k_of_func + b_of_func > float(dataset[i][attr2]):
            l_bottom.append(str(class1) == (dataset[i][-1]))
        elif(float(dataset[i][attr1]) * k_of_func + b_of_func <= float(dataset[i][attr2])):
             l_top.append(str(class2) == str(dataset[i][-1]))
    return (((l_top.count(True)/(len(l_top)+1)+(l_bottom.count(True)/(len(l_bottom)+1))))/2)

'''
test D & E
'''
d1 = dataset[0:110]
d2 = dataset[0:55] + dataset[110:165]
cnt1 = 0
v1 = [-1,5,3]
b1 = [-3,8,6]
while cnt1 < len(d2):
    v1 = train_weightsDE(v1,0.1,d2[cnt1])
    cnt1+=1
    if test_weight(v1,d2,3,4,3,1) > 0.85:
        print("weight value that identifying class3 and class1 (height and width) is: " + str(list(map(r,v1))))
        break
    elif(cnt1 == 109):
        cnt1 = 0

while cnt1 < len(d1):
    b1 = train_weightsDE(b1,0.1,d1[cnt1])
    cnt1+=1
    if test_weight(b1,d1,3,4,1,2) > 0.85:
        print("weight value that identifying class1 and class2 (height and width) is: " + str(list(map(r,b1))))
        break
    elif(cnt1 == 109):
        cnt1 = 0
#generating the weight value based on height and width
##[-97.0, 13.607000000000001, 7.9208] height&width class3&1
##[-111.0, 13.208400000000001, 8.9591] height&width class1&2


'''
Test A & B
'''
d1 = dataset[0:110]
d2 = dataset[0:55] + dataset[110:165]
cnt2 = 0
v2 = [-1,5,3]
b2 = [-3,8,6]
while cnt2 < len(d2):
    v2 = train_weightsAB(v2,1,d2[cnt2])
    if v2[0] < -1300:
        v2 = [-1,5,3]
    cnt2+=1
    if test_weight(v2,d2,0,1,3,1) > 0.80:
        print("weight value that identifying class3 and class1 (area and perimeter) is: " + str(list(map(r,v2))))
        break
    elif(cnt2 == 109):
        cnt2 = 0

while cnt2 < len(d1):
    b2 = train_weightsAB(b2,1,d1[cnt2])
    if b2[0] < -1300:
        b2 = [-3,8,6]
    cnt2 += 1
    if test_weight(b2,d1,0,1,1,2) > 0.80:
        print("weight value that identifying class1 and class2 (area and perimeter) is: " + str(list(map(r,b2))))
        break
    elif(cnt2 == 109):
        cnt2 = 0



#generating the weight value based on area and perimeter
##[-655.789, 40.82, 6.423] attr0 & attr1 class3&1
##[-985.1834999999996, 39.50000000000001, 21.476999999999993] attr0 & attr1 class1&3


'''
Test F & G
'''

d1 = dataset[0:110]
d2 = dataset[0:55] + dataset[110:165]
cnt3 = 0
v3 = [-1,5,3]
b3 = [-3,8,6]
while cnt3 < len(d2):
    v3 = train_weightsFG(v3,1,d2[cnt3])
    if v3[0] < -2700:
        v3 = [-1,5,3]
    cnt3+=1
    if test_weight(v3,d2,5,6,3,1) > 0.80:
        print("weight value that identifying class2 (asymmetry coefficient and length of kernel groove) is: " + str(list((map(r,v3)))))
        break
    elif(cnt3 == 109):
        cnt3 = 0

while cnt3 < len(d1):
    b3 = train_weightsFG(b3,1,d1[cnt3])
    if b3[0] < -2700:
        b3 = [-3,8,6]
    cnt3 += 1
    if test_weight(b3,d1,5,6,1,2) > 0.80:
        print("weight value that identifying class3 and class1 (asymmetry coefficient and length of kernel groove) is: " + str(list(map(r,b3))))
        break
    elif(cnt3 == 109):
        cnt3 = 0
#generating the weight value based on asymmetry coefficient and length of kernel groove       
##[-121.0, -133.74999999999994, 125.72727272727273] attr5&attr6 class3&1
##[-153.0, -2.5, 28.363636363636363] attr5&attr6 class1&2
            
