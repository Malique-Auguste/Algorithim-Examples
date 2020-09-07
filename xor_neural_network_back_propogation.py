import numpy as np

np.random.seed(1)

x_list = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
y_list = np.array([[1],         [1],        [0],        [0]])

'''x_list = np.array([[1, 1, 1], [1, 1, 0]])
y_list = np.array([[1], [0]])'''

parameters = []
'parameters = np.array([[1.25, -5, 4.25]])'
single_change = []
total_change = []

x_propogated = []
z = []
regularization_factor = 0.0005

'number of ouputs'
k = 1
'number of layers'
l = 0
'number of data sets'
m = 7

h = []

def Initialise():
    global x_propogated, m, l, k, total_change, single_change, y_list, x_list, parameters

    'parameters = [np.random.normal(0, 1, size= (4, 3)), np.random.normal(0, 1, size =(1, 5))]'
    parameters = [np.array([[0.4, 0.5, 0.2], [1.1, 1.5, 0.9]]), np.array([[1.1, 1.7, 0.8]])]

    for param in parameters:
        x_propogated.append(np.array(param))
        z.append(np.array(param))
    
    m = len(y_list)
    l = len(parameters)

    k = 1
    for y in y_list:
        if y > k:
            k = y

    '''single_change = np.copy(parameters)
    total_change = np.copy(parameters)'''

def Sigmoid(x, param):
    z = (np.matmul(x,  np.transpose(param)))
    return (1 / (1 + np.exp(np.negative(z))), z)

def Forward_Propogate(x_row):
    global x_propogated, parameters

    (x_propogated[0], z[0]) = Sigmoid(x_row, parameters[0])
    x_propogated[0] = np.hstack((np.ones(1), x_propogated[0]))

    i = 1
    while i < l:
        (x_propogated[i], z[i]) = Sigmoid(x_propogated[i - 1], parameters[i])
        if i != l - 1:
            x_propogated[i] = np.hstack((np.ones(1), x_propogated[i]))
        i += 1

def Regularise_Parameters(params):
    r = 0

    i = 0
    while i < len(params):
        for row in params[i]:
            for x in row:
                r += x ** 2
        i += 1
    
    return r

def Most_Likely(y):
    if y == 1:
        y = 0.999999
    elif y == 0:
        y = 0.000001
    return y

def Loss(params):
    global x_list, y_list, x_propogated, m, l, parameters, h
    i = 0
    h = []
    while i < m:
        Forward_Propogate(x_list[i])
        h.append(Most_Likely(x_propogated[l - 1]))
        i += 1

    h = np.transpose(np.array(h)[np.newaxis])
    
    cost = ((np.matmul(np.negative(y_list).T, np.log(h)) - np.matmul((1 - y_list).T, np.log(1 - h))) / m)
    'cost[0] +=  regularization_factor * Regularise_Parameters(params)'

    return cost[0] 

def Back_Propogation():
    global l, x_list, single_change, total_change, parameters, m
    single_change.clear()
    total_change.clear()
    for param in parameters:
        single_change.append(param)
        total_change.append(np.zeros(param.shape))
    i = 0
    while i < m:
        Forward_Propogate(x_list[i])
        l_ = l - 1
        while l_ > -1:
            if l_ == l - 1:
                single_change[l_] =  x_propogated[l_] - y_list[i]
                '''print("dz2")
                print(single_change[l_])'''
            else:
                sigmoid_derivative = x_propogated[l_] * (1 - x_propogated[l_])
                sigmoid_derivative = np.delete(sigmoid_derivative, 0)
                #print("dz1")
                
                a = np.transpose(parameters[l_ + 1]) * single_change[l_ + 1]
                a = np.delete(a, 0)
                #print(a * sigmoid_derivative)
                single_change[l_] = a * sigmoid_derivative
            l_ -= 1

        l_ = l - 1
        while l_ > -1:
            if l_ == 0:
                #print("dw1")
                x = x_list[i]
                x = np.delete(x_list[i], 0)
                temp = np.hstack(((single_change[l_][..., np.newaxis], single_change[l_][..., np.newaxis] * x)))
                total_change[l_] += temp
                #print(temp)
            else:
                total_change[l_] += np.transpose(single_change[l_]) * x_propogated[l_ - 1]
                '''print("dw2")
                print(np.transpose(single_change[l_]) * x_propogated[l_ - 1])'''
            l_ -= 1
        #print("\n")
            

        i += 1
    '''print("tc")
    print(total_change)'''
    for change in total_change:
        change /= m


Initialise()
print("initialisation complete...\n")
'''
Forward_Propogate(x_list[0])
print(x_propogated[l-1])
print(Loss(parameters))
'''
lowest_loss = 10 ** 20
lowest_iteration = 0
lowest_parameter = []
for param in parameters:
    lowest_parameter.append(param)
    
i = 0
while i < 5000:
    
    if i % 500 == 0:
        print("\nLoss: " + str(Loss(parameters)) + "\titeration: " + str(i))
        print("Parameters:" + str(parameters))
        print("h: " + str(h))

    Back_Propogation()
    
    j = 0
    while j < len(parameters):
        parameters[j] -= 0.1 * total_change[j]
        j += 1

    if Loss(parameters) < lowest_loss:
        lowest_loss = Loss(parameters)
        lowest_iteration = i
        lowest_parameter.clear()
        for param in parameters:
            lowest_parameter.append(param)
    i += 1


print("\n\n\nLowest Loss: " + str(lowest_loss) + "\t iteration: " + str(lowest_iteration))
print("Lowest Parameters:" + str(lowest_parameter))
print("\n")

#print(parameters)
