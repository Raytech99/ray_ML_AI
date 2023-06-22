# 2 layer neural network using numpy
# Table used in program for neural network on page 370
import numpy as np
import tensorflow as tf

# X = input of our 3 input XOR gate
# set up the inputs of the neural network (right from the table)
X = np.array(([0,0,0], [0,0,1], [0,1,0], \
    [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]), dtype=float)
# y = our output of our neural network
y = np.array(([1], [0], [0], [0], [0], \
    [0], [0], [1]), dtype=float)

# what value we want to predict 
xPredicted = np.array(([0,0,1]), dtype=float)

X = X/np.amax(X, axis=0) # maximum of X input array
# maximum of xpredicted (our input data for the predction)
xPredicted = xPredicted/np.amax(xPredicted, axis=0)

# set up our Loss file for graphing

lossFile = open("SumSquaredLossList.csv", "w")

class Neural_Netowrk (object):
    def __init__(self):
        #parameters
        self.inputLeayerSize = 3 # X1,X2,X3
        self.outputLayerSize = 1 # Y1
        self.hiddenLayerSize = 4 # Size of the hidden layer

        # build weights of each layer
        # set to random values 
        # look at the interconnection diagram to make sense of this 
        # 3x4 matrix for input to hidden
        self.W1 = \
            np.random.randn(self.inputLeayerSize, self.hiddenLayerSize)
        # 4x1 matrix for hidden layer to output
        self.W2 = \
            np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
    

    def feedForward(self, X):
        # feedForward propagation through our netwrok 
        # dot product of X (input) and first set of 3x4 weights
        self.z = np.dot(X, self.W1)

        # the activationSigmoid activation function - neural magic
        self.z2 = self.activationSigmoid(self.z)
        # dot product of hidden layer (z2) and second set of 4x1 weights
        self.z3 = np.dot(self.z2, self.W2)

        # final activation function - more nueral magic lol
        o = self.activationSigmoid(self.z3)
        return o


    def backwardPropagate(self, X, y, o):
        # backward propagate through the network
        # calculkate the error output
        self.o_error = y-o

        # apply derivative of activationSigmoid to erroe
        self.o_delta = self.o_error*self.activationSigmoidPrime(o)

        # z2 error: how much our hidden layer wegiths conributed to output 
        # error
        self.z2_error = self.o_delta.dot(self.W2.T)

        # applying derivative to activationSigmoid to z2 error
        self.z2_delta = self.z2_error*self.activationSigmoidPrime(self.z2)

        # adjusting first set (inputLayer --> hiddenLayer) weights
        self.W1 += X.T.dot(self.z2_delta)
        # adjusting second set (hiddenLayer --> output layer) weights
        self.W2 += self.z2.T.dot(self.o_delta)


    def trainNetowrk(self, X, y):
        # feed forward the loop 
        o = self.feedForward(X)
        # and then propagate the values (feedback)
        self.backwardPropagate(X, y, o)


    def activationSigmoid(self, s):
        # activation fucntion
        # simple activationSigmoid curve as in the book
        return 1/(1+np.exp(-s))


    def activationSigmoidPrime(self, s):
        # First derivative of actovationSigmoid
        #calculus time
        return s * (1 - s)


    def saveSumSquaredLossList(self, i, error):
        lossFile.write(str(i)+","+str(error.tolist())+'\n')

    
    def saveWeights(self):
        # save this in order to reproduce the network
        np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

    
    def predictedOutput(self):
        print ("Predicted XOR output data based on trained weights: ")
        print ("Expected (X1-X3): \n" + str(xPredicted))
        print ("Output (Y1) : \n" + str(self.feedForward(xPredicted)))

myNeuralNetowrk = Neural_Netowrk()
trainingEpochs = 1000
#trainingEpochs = 100000

for i in range(trainingEpochs):
    print ("Epoch # " + str(i) + "\n")
    print ("Netowrk Input : \n" + str(X))
    print ("Expected Output from XOR Gate Neural Network: \n" + str(y))
    print ("Actual Output from XOR Gate Neural Network: \n" + \
        str(myNeuralNetowrk.feedForward(X)))
    # mean sum squared loss
    Loss = np.mean(np.square(y - myNeuralNetowrk.feedForward(X)))
    myNeuralNetowrk.saveSumSquaredLossList(i, Loss)
    print ("Sum Squared Loss: \n" +str(Loss))
    print ("\n")
    myNeuralNetowrk.trainNetowrk(X, y)

myNeuralNetowrk.saveWeights()
myNeuralNetowrk.predictedOutput()
