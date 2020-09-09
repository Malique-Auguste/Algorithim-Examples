import numpy as np

#Input and ouput values that the neural network will train on
xList = np.array([[1, 1, 1], [1, 3, 5], [1, 10, 13], [ 1, -2.5, 7]])
yList = np.array([[6], [14.5], [40.5], [6.5]])

#weights
parameters = np.random.normal(0,1,[1,3])

def Normal_Equation():
    parameters= np.matmul(np.linalg.inv(np.matmul(xList.T, xList)) , np.matmul(xList.T , yList))
    return parameters

parameters = Normal_Equation()
print("Parameters:\n" + str(parameters))
while True:
    x_list_input = []
    count = 0
    while count < parameters.shape[0]:
        x_list_input.append(input("Enter x{count} value here: "))

        #If enterd value is not a number the program quits
        try:
            x_list_input[count] = float(x_list_input[count])
        except:
            raise SystemExit("Exiting program...")
        count += 1

    x_list_input = np.array(x_list_input)
    print(f"corresponding y value is: {np.matmul(x_list_input, parameters)}")
