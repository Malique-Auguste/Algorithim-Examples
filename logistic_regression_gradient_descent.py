import numpy as np

#Input and ouput values that the neural network will train on
x_list = np.array([[1, -5, -5], [1, -5, -4], [1, -4, -5], [1, 8, 2], [1, 7, 2], [1, 8, 3], [1, -5, 7], [1, -6, 7], [1, -5, 8]])
y_list = np.array([[0], [0], [0], [1], [1], [1], [2], [2], [2]])


#weights
parameters = np.random.normal(0, 1, (3,3))

max_iterations = 1000
learning_rate = 0.01

#number of ouputs
k = 1
#number of data sets
m = 7

def Initialise():
    global m, k, y_list, parameters

    print("\nBeginning initialisation...")
    
    m = len(y_list)

    k = parameters.shape[0]
  
def Sigmoid(x, parameters):
    z = (np.matmul(x,  np.transpose(parameters)))
    return 1 / (1 + np.exp(np.negative(z)))

def Loss():
    global parameters, x_list, y_list, m, k
    #Calculates the squared error
    cost = []
    class_ = 0
    while class_ < k:
        new_y_list = []
        i = 0
        while i < m:
            new_y_list.append(y_list[i])
            if not y_list[i] == class_:
                new_y_list[i] = 0
            else:
                new_y_list[i] = 1
            i += 1
        new_y_list = np.array(new_y_list)
        h = []
        for x in x_list:
            h.append(Sigmoid(x, parameters[class_]))
        h = np.transpose(np.array(h)[np.newaxis])
        cost.append(np.matmul(np.negative(new_y_list).T, np.log(h)) - np.matmul((1 - new_y_list).T, np.log(1 - h)))
        cost[class_] /= m
        class_ += 1  
    return cost  

def Calculate_Parameter_Error():
    global y_list, x_list, parameters, m
    parameters_error = np.copy(parameters)
    class_ = 0
    while class_ < k:
        new_y_list = []
        i = 0
        while i < m:
            new_y_list.append(y_list[i])
            if not y_list[i] == class_:
                new_y_list[i] = 0
            else:
                new_y_list[i] = 1
            i += 1
        new_y_list = np.array(new_y_list)
        parameters_error[class_] = np.matmul(x_list.T, (Sigmoid(x_list, parameters[class_]) - new_y_list)) / m
        class_ += 1
    return parameters_error

def Gradient_Descent():
    global parameters, max_iterations

    print("Beginning gradient descent...\n")
    
    #if the cost begins to increase the lowest possible cost and its corresponding data is recorded
    lowest_loss = Loss()
    lowest_iteration = 0
    lowest_parameter = np.copy(parameters)

    i = 0
    while i < max_iterations:
        if i % 200 == 0:
            print(f"\nLoss: {Loss()}\tIteration: {i}")
            print("Parameters: " + str(parameters))
        elif i == max_iterations - 1:
            print(f"\nLoss: {Loss()}\tIteration: {i}")
            print(f"Parameters: {parameters}")
    
        parameters -= learning_rate * Calculate_Parameter_Error()

        j = 0
        while j < len(Loss()):
            if Loss()[j] > lowest_loss[j]:
                lowest_loss = Loss()
                lowest_iteration = i
                lowest_parameter = np.copy(parameters)
            j += 1

        i += 1
        
    i = 0
    while i < len(Loss()):
        if Loss()[i] > lowest_loss[i]:
            print("\n\nThe final loss value obtained was greater than the lowest loss obtained.")
            print(f"\nLowest Loss: {lowest_loss}\t iteration: {lowest_iteration}")
            print(f"Lowest Parameters: {lowest_parameter}")
            print("\n")
        i += 1
    
def Test_Parameters():
    global parameters, k
    print("\nEnter anything besides a number to quit.\n")

    while True:
        x_list_input = []

        count=0
        while count < parameters.shape[1]:
            x_list_input.append(input(f"Enter x{count} value here: "))

            #If enterd value is not a number the program quits
            try:
                x_list_input[count] = float(x_list_input[count])
            except:
                raise SystemExit("Exiting program...")
            count += 1

        x_list_input = np.array(x_list_input)
        class_ = 0
        while class_ < k:
            print(f"Probablility of entered values equaling {class_} is: {Sigmoid(x_list_input, parameters[class_])}")
            class_ += 1
        print()


Initialise()
Gradient_Descent()
Test_Parameters()
