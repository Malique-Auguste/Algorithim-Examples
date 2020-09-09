import numpy as np

#Input and ouput values that the neural network will train on
x_list = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
y_list = np.array([[1],         [1],        [0],        [0]])

#weights
parameters = []
#The error calculated for a single x value
small_delta = []
#The error calculated for the entire training data
total_delta = []

#x_propogated is often denoted as the 'activation' values of each layer.
x_propogated = []

#number of ouputs
k = 1
#number of layers
l = 0
#number of data sets
m = 7

learning_rate = 0.1
max_iterations = 10001


def Initialise():
    global x_propogated, m, l, k, total_delta, small_delta, y_list, x_list, parameters

    print("\nBeginning initialisation...")

    #Initialises the parameters of each layer to random values.
    #for any array of paramaters, size = (number of neurons, number of inputs)
    parameters = [np.random.normal(0, 1, size= (2, 3)), np.random.normal(0, 1, size =(1, 3))]

    #Creates the shape of x_propogated from the list of parameters
    for param in parameters:
        x_propogated.append(np.array(param))
    
    m = len(y_list)
    l = len(parameters)

    k = 1
    for y in y_list:
        if y > k:
            k = y
  
def Sigmoid(x, param):
    z = (np.matmul(x,  np.transpose(param)))
    return 1 / (1 + np.exp(np.negative(z)))

def Forward_Propogate(x_row):
    global x_propogated, parameters

    #Calculates the activation value of the first layer
    x_propogated[0] = Sigmoid(x_row, parameters[0])
    #Adds a bias unit to x_propogated (activation) so that it can be immediately used to calculate the activation of the next layer
    x_propogated[0] = np.hstack((np.ones(1), x_propogated[0]))

    i = 1
    while i < l:
        #Calculates the activation value of every subsequent layer
        x_propogated[i] = Sigmoid(x_propogated[i - 1], parameters[i])
        if i != l - 1:
            #If this is not the last activation value(neural network output) a bias unit is added to x_propgated
            x_propogated[i] = np.hstack((np.ones(1), x_propogated[i]))
        i += 1

def Prevent_Error(y):
    #To prevent the divide by zero error in the cost function the vlue is changed by a little bit
    if y == 1:
        y = 0.999999
    elif y == 0:
        y = 0.000001
    return y

def Loss(params):
    global x_list, y_list, x_propogated, m, l, parameters
    #Calculates the total error of the neural network 

    i = 0
    h = []
    while i < m:
        Forward_Propogate(x_list[i])
        h.append(Prevent_Error(x_propogated[l - 1]))
        i += 1

    h = np.transpose(np.array(h)[np.newaxis])
    cost = ((np.matmul(np.negative(y_list).T, np.log(h)) - np.matmul((1 - y_list).T, np.log(1 - h))) / m)

    return cost[0] 

def Back_Propogation():
    global l, x_list, small_delta, total_delta, parameters, m
    small_delta.clear()
    total_delta.clear()

    #Creates the shape of the change varables
    for param in parameters:
        small_delta.append(param)
        total_delta.append(np.zeros(param.shape))

    i = 0
    while i < m:
        #Finds x_propogated for specified x value
        Forward_Propogate(x_list[i])

        l_ = l - 1
        while l_ > -1:
            #if at last layer in neural network the error of the layer is calculated for a specific x-value
            if l_ == l - 1:
                small_delta[l_] =  x_propogated[l_] - y_list[i]
            
            #the error of each subsequent layer is calculated  for a specific x-value
            else:
                
                sigmoid_derivative = x_propogated[l_] * (1 - x_propogated[l_])
                sigmoid_derivative = np.delete(sigmoid_derivative, 0)
                temp = np.transpose(parameters[l_ + 1]) * small_delta[l_ + 1]
                #removing the bias unit from this array
                temp = np.delete(np.transpose(temp), 0, 1)
                if temp.shape[0] == 1:
                    temp = temp[0,...]
                if len(temp.shape) > 1:
                    if temp.shape[0] >= 1:
                        temp = np.sum(temp,0)
                small_delta[l_] = temp * sigmoid_derivative
            l_ -= 1

        l_ = l - 1
        while l_ > -1:
            #The error of the first layer is calculated using the input x values and added to the total error
            if l_ == 0:
                x = x_list[i]
                #Remove bias unit from x 
                x = np.delete(x_list[i], 0)
                total_delta[l_] += np.transpose(small_delta[l_][np.newaxis]) * x_list[i]
            
            #the error of the each other layer is calculated using the propogated x values and added to the total error
            else:
                total_delta[l_] += np.transpose(small_delta[l_][np.newaxis]) * x_propogated[l_ - 1]
            l_ -= 1
            

        i += 1
    
    #At the end of back-propogation the averaged total_delta of all layers are found
    for change in total_delta:
        change /= m

def Gradient_Descent():
    global parameters, max_iterations
    print("Beginning gradient descent...\n")
    
    #if the cost begins to increase the lowest possible cost and its corresponding data is recorded
    lowest_loss = 10 ** 20
    lowest_iteration = 0
    lowest_parameter = []
    for param in parameters:
        lowest_parameter.append(param)
    
    #Back propogates till max_iteration is reached
    i = 0
    while i < max_iterations:
        
        if i % 1000 == 0:
            print(f"\nLoss: {Loss(parameters)}\tIteration: {i}")
            print("Parameters: " + str(parameters))
        elif i == max_iterations - 1:
            print(f"\nLoss: {Loss(parameters)}\tIteration: {i}")
            print(f"Parameters: {parameters}")

        Back_Propogation()
        
        #updates parameters
        j = 0
        while j < len(parameters):
            parameters[j] -= learning_rate * total_delta[j]
            j += 1

        if Loss(parameters) < lowest_loss:
            lowest_loss = Loss(parameters)
            lowest_iteration = i
            lowest_parameter.clear()
            for param in parameters:
                lowest_parameter.append(param)
        i += 1

    if Loss(parameters) > lowest_loss:
        print("\n\nThe final loss value obtained was greater than the lowest loss obtained.")
        print(f"\nLowest Loss: {lowest_loss}\t iteration: {lowest_iteration}")
        print(f"Lowest Parameters: {lowest_parameter}")
        print("\n")
    
def TestNeuron():
    global parameters
    print("\nEnter anything besides a number to quit.\n")

    while True:
        x_list_input = []

        count=0
        while count < len(parameters[0][0]):
            x_list_input.append(input(f"Enter x{count} value here: "))

            #If enterd value is not a number the program quits
            try:
                x_list_input[count] = float(x_list_input[count])
            except:
                raise SystemExit("Exiting program...")
            count += 1

        x_list_input = np.array(x_list_input)
        Forward_Propogate(x_list_input)
        print(f"Probablility of entered values equaling 1 is: {x_propogated[l - 1]}")
        print()



Initialise()
Gradient_Descent()
TestNeuron()
