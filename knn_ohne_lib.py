
import numpy
# scipy.special for the sigmoid function expit()
import scipy
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
import imageio
import cv2
import os
import glob
import csv
import time

import scipy
from scipy import special

# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
                                            ## zu 1/(1+np.exp(-X)) ändern wenns funktioniert
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

            
def make_dataset():
    train_list_path=[]
    test_test_path =[]

    dir_list = os.listdir("./dataset/dataset_simple")
    for class_dir in dir_list:
        if class_dir == "fork":
            class_num = 0
        elif class_dir == "spoon":
            class_num = 1
        elif class_dir == "knife":
            class_num = 2
        pic_list = glob.glob("./dataset/dataset_simple"+"/"+class_dir+"/"+"*.jpg")
        #pic_list = glob.glob("./dataset/dataset_blackandwhite"+"/"+class_dir+"/"+"*.png")
        i = 0
        for line in pic_list:
            string = str(class_num)+","+line[:-4]+",\n"
            if i % 3:
                train_list_path.append(string)
            else:
                test_test_path.append(string)
            i += 1
    #print("train_list_path",len(train_list_path), train_list_path)
    #print("test_test_path",len(test_test_path), test_test_path)
    return train_list_path, test_test_path

    


if __name__ == "__main__":

    training_data_list,test_data_list = make_dataset()

    # number of input, hidden and output nodes
    input_nodes = 40000
    hidden_nodes = 8000
    output_nodes = 3

    # learning rate
    learning_rate = 0.01

    # create instance of neural network
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    # load the mnist training data CSV file into a list
    #training_data_file = open("train.csv", 'r')
    #training_data_list = training_data_file.readlines()
    #training_data_file.close()

    # train the neural network



    # epochs is the number of times the training data set is used for training
    epochs = 50

    for e in range(epochs):
        # go through all records in the training data set
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = cv2.imread(all_values[1]+".jpg",0)
            inputs = cv2.resize(inputs,(200,200))
            inputs = inputs.reshape(40000)
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            print("training")
            n.train(inputs, targets)
            pass
        print("Epoche:",e)
        if e == 10:
            print(True)


    # test the neural network

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = cv2.imread(all_values[1]+".jpg",0)
        inputs = cv2.resize(inputs,(200,200))
        inputs = inputs.reshape(40000)
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        print(outputs)
        label = numpy.argmax(outputs)
        print("label",label)
        print("correct_label",correct_label)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
        pass


    

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)






    # test the neural network with our own images

    # load image data from png files into an array
    # unkomment if u wanna see a test img

    #test_img = cv2.imread("test_spoon.jpg",0)
    #test_img = cv2.resize(test_img,(40,40))
    #img_data = test_img.reshape(1600)
    #test_img = cv2.resize(test_img,(200,200))
    #cv2.imshow("img_img",test_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()'
    #outputs = n.query(img_data)
    #print(outputs)
    #label = numpy.argmax(outputs)
    #label_lookup = {0:"Gabel",1:"Löffel",2:"Messer"}
    #print("network says ", label_lookup[label])

    pass