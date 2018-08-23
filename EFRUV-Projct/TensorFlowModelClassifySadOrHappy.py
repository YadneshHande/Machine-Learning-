#! /usr/bin/env python

"""
This programming model is used to classify sad or happy phases. 
To achieve this labelled pictures are loaded into our model. 
These are then transformed into Numpy array dataSet with their labels accordingly. So we get a TrainingSet and a TestingSet
After the testingSet and TrainingSet are created with their sets of labels as well, the TensorFlowModel is executed.
In the tensorflow model the actual learning process starts to happen. 
We have a placeholder for the numpy images and a placeholder for the labels.
We use a convolutional layer, a Relu, a max pooling layer and we use in the end Dropout. 
Then we apply Softmax and cross entropy
We use the adamOptimizer to learn with a certain learning rate and we minimize with the cross entropy
Then we iterate in batches through the whole trainingset. Each batch get's trained and there is a test batch.
The results are then plotted in Graphs to visualize the accuracy and loss of the testingSet and the trainingSet
"""


import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random
import matplotlib
matplotlib.use('Agg') #To use matplotlib in headless mode
import matplotlib.pyplot as plt

def transformImage(file):
    """Transforming a image into a numpy array
    params: The image file
    Returns the transformed image as numpy array in the dimension (widht, length, color)"""

    im = Image.open(file)
    pix = im.load()
    width = im.size[0]
    height = im.size[1]

    #store the rgb information in a numpy array (width, height, color).
    picture = np.array([[pix[x,y] for y in range(height)] for x in range(width)], np.int32)

    return picture


def createDataSets(smilePath, nonSmilePath, dataSetSize, testingSplit):
    """Createts the training and test datasets from the images in smilePath and nonSmilePath.
    Params: The path the the smiling images, the path to the non smiling images, the size of the dataSet we want to use
    The split value for the testing and training set. The split is set based on the testing split size in percent.
    If a testing split of 20 is chosen 20% are going to be test data and 80% training data.
    Returns: the testing and training labels, the trainingSet and TestingSet
    """

    trainingLabels = []
    trainingSetFiles = []
    testingLabels = []
    testingSet = []

    # transform all smiling pictures
    for root, dirs, files in os.walk(smilePath, True):
        i=0
        #static for loop
        for name in files:
        #all images
        #for name in files:
            if name.endswith(".jpg") and (i<(dataSetSize/2) or dataSetSize == -1):
                if random.randint(1, 100) > testingSplit:
                    trainingSetFiles.append(os.path.join(root, name))
                    trainingLabels.append(np.array([1,0], np.int32))
                else:
                    testingSet.append(transformImage(os.path.join(root, name)))
                    testingLabels.append(np.array([1,0], np.int32))
                i=i+1

    # transform all non-smiling pictures
    #the non smiling pictures are added to a random position in the trainingSet and labels and the testingSet and labels
    #the sets and labelled where already created in the above for loop. 
    for root, dirs, files in os.walk(nonSmilePath, True):
        k=0
        #all images
        #for name in files:
        #static for loop
        for name in files:
            if name.endswith(".jpg") and (k<(dataSetSize/2) or dataSetSize == -1):
                if random.randint(1, 100) > testingSplit:
                    # insert to a random position to avoid overfitting
                    insertPosition = random.randint(0, len(trainingLabels))
                    trainingSetFiles.insert(insertPosition, os.path.join(root, name))
                    trainingLabels.insert(insertPosition, np.array([0, 1], np.int32))
                else:
                    # insert to a random position to avoid overfitting
                    insertPosition = random.randint(0, len(trainingLabels))
                    testingSet.insert(insertPosition, transformImage(os.path.join(root, name)))
                    testingLabels.insert(insertPosition, np.array([0, 1], np.int32))
                k=k+1

    return trainingSetFiles,trainingLabels,testingSet,testingLabels
    #TODO: Needs to be explained better Side note: Only the file names of the training images are provided to reduce memory consumption.


def tensorFlowModel(trainingSet,trainingLabels,testingSet,testingLabels,batchSize):
    """The actual tensorflow Model
    Here we initialise first all the variables, such as the numpy images, the labels, the learning rate, the weights, the bayes. 
    We then initialize a convolutional layer and initialize ReLu and max pooling
    Then we reshape it for the processing
    We perform a dropout to make the analysis perform faster and prevent overfitting
    The model will then be trained in batches.
    Each batch has a TrainingSet and a testingSet, For each training Image an iteration will be performed
    The training session uses the dropout and will be trained, the testing session not. 
    After all the batches are finished there will be a train result for the accuracy and the loss. 
    There is also a testing result for the accuracy and the loss
    Params: The trainingSet, trainingLabels, the TestingLabels and the TestingSet, The batch size the model will be trained in
    Returns: the training accuracy, the training loss, the testing accuracy, the testing loss
    """
    #placeholder for the numpy images
    X = tf.placeholder(tf.float32, [None, 320, 240, 3])
    #placeholder for the labels
    Y_ = tf.placeholder(tf.float32, [None, 2])
    #placeholder for the learning rate. 
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # weight 1
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 12], stddev=0.1))
    # bias
    B1 = tf.Variable(tf.ones([12]) / 3)
    # convolutional layer 1
    Y1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    #Initialize the Relu layer
    Y1Relu = tf.nn.relu(Y1 + B1)
    # max layer pooling
    pool1 = tf.nn.max_pool(Y1Relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')  # halve structure to 160x120

    # reshape
    YY = tf.reshape(pool1, shape=[-1, 120 * 160 * 12])  # 160*120 structure
    # Apply Dropout
    YY = tf.nn.dropout(YY, keep_prob)

    # weight 2
    W2 = tf.Variable(tf.truncated_normal([160 * 120 * 12, 2], stddev=0.1))
    # bias
    B2 = tf.Variable(tf.ones([2]) / 3)

    # softmax
    Ylogits = tf.matmul(YY, W2) + B2
    Y = tf.nn.softmax(Ylogits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)
    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # set learning rate and learning rate decay
    initial_learning_rate = 0.0005
    decay_rate = 0.95
    decay_steps = 500
    stepCounter = 0

    # set dropout keep probability
    dropout = 0.80

    # sets to plot
    train_a = []
    train_c = []
    test_a = []
    test_c = []

    # train the model in batches
    for step in range(0,len(trainingSet),batchSize):

        #reduce the learning rate
        if stepCounter > decay_steps:
            lr = initial_learning_rate * decay_rate ** (int(stepCounter / decay_steps))
        else:
            lr = initial_learning_rate

        # use the next batch
        batchBegin = step
        batchEnd = step+batchSize
        if batchEnd > len(trainingSet):
            batchEnd = len(trainingSet)

        # generate the next batch
        nextBatch = []
        for image in trainingSet[batchBegin:batchEnd]:
            nextBatch.append(transformImage(image))
        batch_X = np.asarray(nextBatch)
        del nextBatch[:]
        batch_Y = np.asarray(trainingLabels[batchBegin:batchEnd])

        # train
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, learning_rate:lr, keep_prob:dropout})
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, keep_prob:1.0})
        train_a.append(a)
        train_c.append(c)
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: testingSet, Y_: testingLabels, keep_prob:1.0})
        test_a.append(a)
        test_c.append(c)

        stepCounter += 1

    return train_a, train_c, test_a, test_c


def plotResults(train_a, test_a, train_c, test_c):
    """
    Plot and visualise the accuracy and loss
    accuracy training vs testing dataset
    Params: the training accuracy, testing accuracy, training loss, testing loss
    Returns the plot for the accuracy and the loss as an image
    """
    plt.plot(train_a, label='training dataset')
    plt.plot(test_a, label='test dataset')
    plt.legend(bbox_to_anchor=(0, 0.95), loc='lower left', ncol=1)
    plt.xlabel('# batch')
    plt.ylabel('accuracy')
    plt.grid(True)
    #plt.show()
    plt.savefig('accuracy.png')
    plt.clf()

    # loss training vs testing dataset
    plt.plot(train_c, label='training dataset')
    plt.plot(test_c, label='test dataset')
    plt.legend(bbox_to_anchor=(0, 0.95), loc='lower left', ncol=1)
    plt.xlabel('# batch')
    plt.ylabel('loss')
    plt.grid(True)
    #plt.show()
    plt.savefig('loss.png')


def main(argv=None):
    """
    In the main function we initialize the datasetSize, the testing split and the batch size
    The DataSetSize says how much images we of each set want to use, for example of happy pictues and non happy pictures
    The batchsize defines the size used for training
    When this is initialized we create the Datasets, here we obtain the training set and the testing set
    We run our tensorflow model
    We then visualize the outcome of our tensorflow model, by plotting the result
    Returns: The plots created in the plotResults function. 
    """
    dataSetSize = 750 # use -1 for all images
    testingSplit = 20 # in % of total data-set size
    batchSize = 25
    trainingSetFiles, trainingLabels, testingSet, testingLabels = createDataSets("AMFED/AMFED/happiness/","AMFED/AMFED/nonHappiness/",dataSetSize,testingSplit)
    #batchSize = len(testingSet) #to train in batches of the testing set size
    print "size of training set:", len(trainingSetFiles), len(trainingLabels)
    print "size of testing set:", len(testingSet), len(testingLabels)
    train_a, train_c, test_a, test_c = tensorFlowModel(trainingSetFiles,trainingLabels,testingSet,testingLabels,batchSize)
    #print "Training and Testing - Accurracy, Cross Entropy:"
    #print train_a, train_c
    #print test_a, test_c
    plotResults(train_a, test_a, train_c, test_c)

if __name__ == '__main__':
    main()