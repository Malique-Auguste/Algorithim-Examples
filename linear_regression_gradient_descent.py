import numpy as np

#Input and ouput values that the neural network will train on
x_list = np.array([[1, 2, 4], [1, 4, 2], [1, -5, 7], [1, 2, 9], [1, 7,-3]])
y_list = np.array([[8.25], [-0.75], [26.75], [24.5], [-20.75]])

#weights
parameters = np.random.normal(0, 1, (1,3))

#number of data sets
m = 0

learning_rate = 0.01
lambada = 0.0005
max_iterations = 2000

def Initialise():
    global m
    m = y_list.shape[0]

def Calculate_Y(x):
    global parameters
    return np.matmul(x, np.transpose(parameters))

def Regularise_Parameters(parameters, sum_ = True):
    global lambada

    if not sum_:
        return parameters * lambada

    r = 0
    for parameter in parameters:
        i = 0
        while i < parameter.shape[0]:
            if len(parameter.shape) > 1:
                j = 0
                while j < parameter.shape[1]:
                    r += parameter[i,j] ** 2
                    j += 1
            else:
                r += parameter[i] ** 2
            i += 1
    return r * lambada

def Loss():
    global parameters, x_list, y_list, m
    #Calculates the squared error
    
    cost = 0
    i = 0
    
    while i < m:
        cost += (y_list[i] - Calculate_Y(x_list[i])) ** 2 + Regularise_Parameters(parameters)
        i += 1
    return cost / m
        
def Calculate_Parameter_Error():
    global parameters, x_list, y_list
    return np.transpose(np.matmul(np.transpose(x_list) , (Calculate_Y(x_list) - y_list)))

def Gradient_Descent():
    global parameters, max_iterations

    print("Beginning gradient descent...\n")
    
    #if the cost begins to increase the lowest possible cost and its corresponding data is recorded
    lowest_loss = 10 ** 20
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
    
        parameters -= learning_rate * Calculate_Parameter_Error() + Regularise_Parameters(parameters, False)

        if Loss() < lowest_loss:
            lowest_loss = Loss()
            lowest_iteration = i
            lowest_parameter = np.copy(parameters)

        i += 1
        
    if Loss() > lowest_loss:
        print("\n\nThe final loss value obtained was greater than the lowest loss obtained.")
        print(f"\nLowest Loss: {lowest_loss}\t iteration: {lowest_iteration}")
        print(f"Lowest Parameters: {lowest_parameter}")
        print("\n")

def Test_Parameters():
    global parameters
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
        print(f"Probablility of entered values equaling 1 is: {Calculate_Y(x_list_input)}")
        print()

Initialise()
print(Regularise_Parameters(parameters))
Gradient_Descent()
print(Regularise_Parameters(parameters))
Test_Parameters()
