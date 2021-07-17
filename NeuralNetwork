import numpy as np, random, math

weight1 = [random.random(), random.random(), random.random()] #weights for the first neuron in the first preceptron layer
weight2 = [random.random(), random.random(), random.random()] #weights for the second neuron in the first preceptron layer
weight3 = [random.random(), random.random(), random.random()] #weights for the only neuron in the second preceptron layer
                                    #note the last weight is the bias
lr = 0.001

def lossFunction(pV, aV): #Loss function to determine the optimization algorithm
    if aV == 1:
        return -math.log(pV) #**** this function is not used in the actual program, however, helps visualize how the optimization algorithm works (derivative of the loss function)
    else:
        return -math.log(1 - pV)

def activationFunction(x): #sigmoid activation function for a binary classification (0 or 1)
    b = (1 / (1 + np.exp(-x)))
    return b

def preceptron2(x1, x2, rA): #second precpetron layer (output layer) that joins the two outputs of the previous layer into one
    outputP = x1 * weight3[0] + x2 * weight3[1] + (weight3[2])
    outputP = activationFunction(outputP)

    print(outputP, rA)
    #error = rA - outputP **note this is for debugging purposes
    if rA == 1:
        gradient = -1/(outputP + 0.000000000015)

        weight3[2] = weight3[2] - gradient * lr

        weight3[0] = weight3[0] - gradient*lr
        weight3[1] = weight3[1] - gradient * lr

    else:
        gradient = 1/(1 - outputP + 0.0000000015)
        weight3[2] = weight3[2] - gradient * lr

        weight3[0] = weight3[0] - gradient * lr
        weight3[1] = weight3[1] - gradient * lr
    #print("ERROR ", rA - outputP) **debugging purposes 
    #return (outputP) debugging purposes

def testpreceptron2(x1, x2): #used for testing the neural network
    outputP = x1 * weight3[0] + x2 * weight3[1] + (weight3[2])
    outputP = activationFunction(outputP)
    return outputP

def testPreceptron1(x1, x2): #used for testing the nerual network
    output1 = x1 * weight1[0] + x2 * weight1[1] + weight1[2]
    output2 = x1 * weight2[0] + x2 * weight2[1] + weight2[2]

    output1 = activationFunction(output1)
    output2 = activationFunction(output2)
    # print(output1, output2)
    return (testpreceptron2(output1, output2))

def preceptron1(x1, x2, rA): #first precptron layer, I used the Cross-Entropy cost function, with the Gradient Descent optimization algorithm
    output1 = x1 * weight1[0] + x2 * weight1[1] + weight1[2]
    output2 = x1 * weight2[0] + x2 * weight2[1] + weight2[2]

    output1 = activationFunction(output1)
    output2 = activationFunction(output2)

    #print(output1, output2)
    preceptron2(output1, output2, rA)
    if rA == 1:
        gradient1v1 = -1/(output1 + 0.00000015)
        gradient1v2 = -1/(output2 + 0.00000015)

        weight1[2] = weight1[2] - gradient1v1 * lr #biases
        weight2[2] = weight2[2] - gradient1v2 * lr

        weight1[0] = weight1[0] - gradient1v1 * lr
        weight1[1] = weight1[1] - gradient1v1 * lr #weights
        weight2[0] = weight2[0] - gradient1v2 * lr
        weight2[1] = weight2[1] - gradient1v2 * lr
    else:
        gradient2v1 = 1/(1 - output1 + 0.000000015)
        gradient2v2 = 1/(1 - output2 + 0.000000015)

        weight1[2] = weight1[2] - gradient2v1 * lr  # biases
        weight2[2] = weight2[2] - gradient2v2 * lr

        weight1[0] = weight1[0] - gradient2v1 * lr
        weight1[1] = weight1[1] - gradient2v1 * lr
        weight2[0] = weight2[0] - gradient2v2 * lr
        weight2[1] = weight2[1] - gradient2v2 * lr
    #return (preceptron2(output1, output2, rA))

def main():
    for i in range(10): #trains the nerual network (10 train sets for each combination of inputs)
        preceptron1(1, 1, 0)
        preceptron1(1, 0, 1)
        preceptron1(0, 1, 1)
        preceptron1(0, 0, 0)

    x = int(input())
    y = int(input())
    while x != 2: #allows for user input
        output1 = x * weight1[0] + y * weight1[1] + weight1[2]
        output2 = x * weight2[0] + y * weight2[1] + weight2[2]
        output1 = activationFunction(output1)
        output2 = activationFunction(output2)
        outputF = output1 * weight3[0] + output2 * weight3[1] + weight3[2]
        outputF = activationFunction(outputF)

        if outputF < testPreceptron1(1, 1) and outputF > testPreceptron1(0, 0):
            outputF = 1
        else:
            outputF = 0

        print(outputF)
        x = int(input())
        y = int(input())
main()
