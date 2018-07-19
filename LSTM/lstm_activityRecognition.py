from __future__ import print_function
import collections
import os
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import argparse
import pdb
import json
import random
from network import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import shutil
import pathlib


###CONFUSIONMATRIX###
def plot_cont(true, pred):
    y = []
    p = []
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    def update(i):
        pi = pred[i]
        yi = true[i]
        y.append(yi)
        p.append(pi)
        ax.clear()
        cm = confusion_matrix(y, p)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
        tick_marks = np.arange(len(activity))
        # plt.xticks(tick_marks, activity, rotation=45)
        plt.yticks(tick_marks, activity)

    a = anim.FuncAnimation(fig, update, frames=len(pred), repeat=False)
    plt.show()
#####################

def returnID(Name):
    if(Name == 'Lawn'):
        return 0
    if(Name == 'Camel'):
        return 1
    if(Name == 'Rope'):
        return 2
    if(Name == 'Sky'):
        return 3
    if(Name == 'Cowboy Hat'):
        return 4
    if(Name == 'Hand'):
        return 5
    if(Name == 'nail_clipper'):
        return 6
    if(Name == 'scissor'):
        return 7
    if(Name == 'Horse saddle'):
        return 8
    if(Name == 'Horse bridle'):
        return 9
    if(Name == 'Bull'):
        return 10
    if(Name == 'Brush'):
        return 11
    if(Name == 'Spear'):
        return 12
    if(Name == 'Calf'):
        return 13
    if(Name == 'Shower_handle'):
        return 14
    if(Name == 'Water'):
        return 15
    if(Name == 'Red fabric'):
        return 16
    if(Name == 'Riding Hat'):
        return 17
    if(Name == 'Polo stick'):
        return 18
    if(Name == 'Shaver'):
        return 19
    if(Name == 'horseshoe'):
        return 20
    if(Name == 'Dog Leash'):
    # Name of classes deriving from COCO
        return 21
    if(Name == 'Person'):
        return 22
    if(Name == 'car'):
        return 23
    if(Name == 'traffic light'):
        return 24
    if(Name == 'cat'):
        return 25
    if(Name == 'dog'):
        return 26
    if(Name == 'cow'):
        return 27
    if(Name == 'horse'):
        return 28
    if(Name == 'backpack'):
        return 29
    if(Name == 'frisbee'):
        return 30
    if(Name == 'suitcase'):
        return 31
    if(Name == 'sports ball'):
        return 32
    if(Name == 'couch'):
        return 33
    if(Name == 'scissor'):
        return 34
    if(Name == 'toothbrush'):
        return 35
    if(Name == 'hair_drier'):
        return 36
    if(Name == 'handbag'):
        return 37
    if(Name == 'bench'):
        return 38
    if(Name == 'sink'):
        return 39
    if(Name == 'bowl'):
        return 40
    if(Name == 'stop sign'):
        return 41
    #put activity
    if(Name == 'Walking the dog'):
        return 0
    if(Name == 'Grooming dog'):
        return 1
    if(Name == 'Grooming horse'):
        return 2
    if(Name == 'Clipping cat claws'):
        return 3
    if(Name == 'Bullfighting'):
        return 4
    if(Name == 'Calf roping'):
        return 5
    if(Name == 'Bathing dog'):
        return 6
    if(Name == 'Playing polo'):
        return 7
    if(Name == 'Disc dog'):
        return 8
    if(Name == 'Camel ride'):
        return 9
    if(Name == 'Horseback riding'):
        return 10

    #.....
def reverse(vector):
    if(np.array_equal(vector,[np.array([1,0,0,0,0,0,0,0,0,0,0])])):
        return 0
    if(np.array_equal(vector,[np.array([0,1,0,0,0,0,0,0,0,0,0])])):
        return 1
    if(np.array_equal(vector,[np.array([0,0,1,0,0,0,0,0,0,0,0])])):
        return 2
    if(np.array_equal(vector,[np.array([0,0,0,1,0,0,0,0,0,0,0])])):
        return 3
    if(np.array_equal(vector,[np.array([0,0,0,0,1,0,0,0,0,0,0])])):
        return 4
    if(np.array_equal(vector,[np.array([0,0,0,0,0,1,0,0,0,0,0])])):
        return 5
    if(np.array_equal(vector,[np.array([0,0,0,0,0,0,1,0,0,0,0])])):
        return 6
    if(np.array_equal(vector,[np.array([0,0,0,0,0,0,0,1,0,0,0])])):
        return 7
    if(np.array_equal(vector,[np.array([0,0,0,0,0,0,0,0,1,0,0])])):
        return 8
    if(np.array_equal(vector,[np.array([0,0,0,0,0,0,0,0,0,1,0])])):
        return 9
    if(np.array_equal(vector,[np.array([0,0,0,0,0,0,0,0,0,0,1])])):
        return 10

parser = argparse.ArgumentParser()
parser.add_argument('--run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument("--batch_size", type=int, default=128, help="Desired batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="Desired number of epochs")
parser.add_argument("--sequence_length", type=int, default=8, help="Desired sequence_length")
args = parser.parse_args()

#define class and activity
classes = ['Riding hat','Polo stick','Cowboy Hat','Calf','Rope','Lawn','Sky','Brush','Shaver','Spear','Bull','Red fabric','scissor','nail_clipper','horseshoe','Dog Leash','Horse bridle','Horse saddle','Camel','Hand','Water','Shower_handle','person', 'car', 'traffic light' ,'cat', 'dog', 'horse', 'cow','backpack','frisbee','suitcase','sports ball','couch','scissors','toothbrush','hair drier','handbag','bench','sink','bowl','stop sign']
activity = ['Walking_the_dog','Grooming_dog','Grooming_horse','Clipping_cat_claws','Bullfight','Calf_roping','Bathing_dog','Playing_polo','Disc_dog','Camel_ride','Horseback_riding']
num_classes = 42
num_activity = 11
old_entries = []

try:
    if args.run_opt == 1:
        with open("./Dataset_LSTM/train.json", "r") as fout:
            old_entries = json.load(fout)
    if args.run_opt == 2:
        with open("./Dataset_LSTM/evaluation.json", "r") as fout:
            old_entries = json.load(fout)

except FileNotFoundError:
    old_entries = {}



###READ DATA###
b = old_entries

all_x = []
all_y = []
to_shuffle = []
for xx in b.keys():
    to_shuffle.append(xx)
random.shuffle(to_shuffle)

for xx in to_shuffle:
#for xx in b.keys():

    #temp = b[xx][0]['masks']
    temp = b[xx]['masks']
    #temp_2 = []
    to_insert = True
    counter = 0
    temp_single_video = []
    for yy in range(len(temp)):
        temp_2 = []
        #temp_3 = b[xx][0]['masks'][yy]
        temp_3 = b[xx]['masks'][yy]
        for zz in range(len(temp_3)):

            eliminate_none = returnID(temp_3[zz])
            if(len(temp_3[zz]) == 0):
                continue
            eliminate_none
            temp_2.append(returnID(temp_3[zz]))
        if(len(temp_2) == 0):
            to_insert = False
            continue
        counter = counter +1
        temp_single_video.append(temp_2)
    all_x.append(temp_single_video)
    temp = b[xx]['activity']
    if(to_insert):
        all_y.append(returnID(temp))
####



#####GENERATE BATCH#####
class BatchGenerator(object):

    def __init__(self, data_x,data_y, num_steps, batch_size, vocabulary_1, vocabulary_2,train):
        self.data_x = data_x
        self.data_y = data_y
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary_1 = vocabulary_1
        self.vocabulary_2 = vocabulary_2
        self.current_idx = 0
        #self.skip_step = skip_step
        self.train = train

    def generate(self):
        #print("#############")
        x = np.zeros((self.batch_size, self.num_steps, self.vocabulary_1))
        y = np.zeros((self.batch_size, self.vocabulary_2))
        while True:
            lenght = np.full((self.batch_size), num_steps)
            for i in range(self.batch_size):
                if self.current_idx >= len(self.data_x):
                    if(not self.train):
                        return False
                    self.current_idx = 0

                temp_x = self.data_x[self.current_idx]
                temp_y = self.data_y[self.current_idx]

                effective_lenght = len(temp_x) + 1
                lenght[i] = effective_lenght

                x_final = []
                y_final = []
                condition_y = temp_y#[0]
                to_skip = 0
                for j in range(0,self.num_steps):
                    x_sum = [np.zeros(42)]
                    if(j < len(temp_x)):
                        #x_temp = to_categorical(temp_x[j],num_classes=self.vocabulary_1)
                        x_temp = np.eye(self.vocabulary_1)[temp_x[j]]
                        #with self.sess.as_default():
                        #    x_temp = tf.one_hot(temp_x[j],1).eval()
                        #to_skip = len(x_temp)
                        #print("X_temporaneo")
                        #print(x_temp)
                        #x_sum = [np.zeros(42)]
                        for k in range(0,len(x_temp)):
                            x_sum = x_sum + x_temp[k]
                        x_final.append(x_sum)
                        #y_final = to_categorical(temp_y[j], num_classes=self.vocabulary_2)
                        #y_final.append(y_temp)
                    else:
                        x_final.append(x_sum)
                y_final = np.eye(self.vocabulary_2)[temp_y]
                #y_final = to_categorical(temp_y, num_classes=self.vocabulary_2)
                x_single = np.asarray(x_final)
                y_single = np.asarray(y_final)
                x_single = x_single.reshape(1,self.num_steps,42)
                y_single = y_single.reshape(1,1,11)


                c = x_single.shape
                if(len(c) == 2):
                    x_single = x_single.reshape(1,num_steps,42)


                #y = y.reshape(1,num_steps,11)
                #lenght = np.full((1), num_steps)
                #self.current_idx += x.shape[1]
                #self.current_idx += self.skip_step
                x[i] = x_single
                y[i] = y_single
                self.current_idx += 1

            if(self.train):
                self.data_x, self.data_y = shuffle(self.data_x, self.data_y)
            yield x, y, lenght
###########


num_steps = args.sequence_length # sequence_length
batch_size = args.batch_size
num_epochs = args.num_epochs
size_net = 50
dropout_rate = 0.9
print("NUMERO CLASSI")
print(num_classes)



path_lstm_weights = "saved_networks_lstm_batch{}_epochs{}_seqL_{}".format(batch_size, num_epochs, num_steps)
###TRAIN###
if args.run_opt == 1:
    net = network(num_steps,num_classes,11,size_net, keep_prob = dropout_rate)
    if os.path.exists(path_lstm_weights):
        shutil.rmtree(path_lstm_weights, ignore_errors=False, onerror=None)
    os.makedirs(path_lstm_weights)
###TEST###
if args.run_opt == 2:
    net = network(num_steps,num_classes,11,size_net, keep_prob = 1.0)


###SESSION###
session = tf.Session()
saver = tf.train.Saver()
session.run(tf.global_variables_initializer())
checkpoint = tf.train.get_checkpoint_state(path_lstm_weights)
##############

###TORESTORE###
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(session, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights -- NEW TRAINING")
###############

###TRAIN###
if args.run_opt == 1:
    train_data_generator = BatchGenerator(all_x,all_y, num_steps, batch_size, num_classes, num_activity,True)
    writer = tf.summary.FileWriter(path_lstm_weights)
    summ = tf.summary.merge_all()
    for i in range(0,num_epochs):
        data = next(train_data_generator.generate())

        #lenght = np.full((batch_size), num_steps)
        lenght = data[2]

        loss = session.run(net.loss, feed_dict={net.input: data[0],net.labels: data[1],net.lenghts: lenght})

        accuracy = session.run(net.correct_classified, feed_dict={net.input: data[0],net.labels: data[1],net.lenghts: lenght})
        s = session.run(summ, feed_dict={net.input: data[0],net.labels: data[1],net.lenghts: lenght})

        writer.add_summary(s, i)
        print("\r Epoch : {} -- Loss : {} -- Accuracy : {}".format(i+1, loss, accuracy), end= " ")
        _ = session.run(net.train_step, feed_dict={net.input: data[0],net.labels: data[1],net.lenghts: lenght})
    saver.save(session, path_lstm_weights + '/-lstm')
    print('\nRun `tensorboard --logdir=%s` to see the results.' % path_lstm_weights)

###TEST###
elif args.run_opt == 2:
    example_training_generator = BatchGenerator(all_x,all_y, num_steps, 1, num_classes, num_activity,False)

    num_predict = 2000
    success = 0
    success_second =  0
    success_third = 0
    booo = 0
    iteration = 0
    predictions_vector = []
    reals_vector = []
    #lenght = np.full((1), num_steps)
    lenght = 0
    for i in range(num_predict):
        try:
            data = next(example_training_generator.generate())
            lenght = data[2]

        except StopIteration:
            print("Fine")
            break
        except Exception as e:
            #print("FINE")
            break
        iteration = iteration +1
        prediction = session.run(net.softmax, feed_dict={net.input: data[0],net.labels: data[1],net.lenghts: data[2]})

        predict_word = np.argmax(prediction)
        sorted_value = np.argsort(prediction)

        position = sorted_value[0][9]

        second_value = position
        third_value = sorted_value[0][8]


        y_truth = reverse(data[1])
        if(y_truth == 7):
            booo = booo + 1
        predictions_vector.append(predict_word)
        reals_vector.append(y_truth)
        if(predict_word == y_truth):
            success = success + 1
        if(predict_word == y_truth or second_value == y_truth):
            success_second = success_second + 1
        if(predict_word == y_truth or second_value == y_truth or third_value == y_truth):
            success_third = success_third + 1
    print("#############")
    plot_cont(reals_vector, predictions_vector)
    print("percentuale primo valore: " + str(success/iteration))
    print("percentuale second_value: " + str(success_second/iteration))
    print("percentuale third_value: " + str(success_third/iteration))
