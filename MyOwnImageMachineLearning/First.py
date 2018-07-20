from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#########################################################################
#-------------------------------Notes-----------------------------------#
#------------------See below for source of this project:----------------#
#-https://becominghuman.ai/transfer-learning-retraining-inception-v3-for-custom-image-classification-2820f653c557#
#----------------------------Or his git:--------------------------------# 
#----https://github.com/wisdal/Image-classification-transfer-learning---#
#---------------Thank you Wisdom d'Almeida for this project-------------#
#########################################################################
'''
#########################################################################
#------------------------------------1----------------------------------#
#--------------------------Training the system--------------------------#
#########################################################################

#########################################################################
#-----------------------------------1.0---------------------------------#
#---------------------------------Imports-------------------------------#
#########################################################################
import os,sys
import h5py
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm import tqdm
from PIL import Image

#########################################################################
#-----------------------------------1.1---------------------------------#
#------------------Source and labels. Used throughout-------------------#
#########################################################################
#directions to Sources. Change if want different training data
data_root = './Data/'
train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')
#Finding labels for each item
label_counts = train.label.value_counts()
print(label_counts)

#########################################################################
#-----------------------------------1.2---------------------------------#
#-------------------------Separation of images--------------------------#
#########################################################################
#Sorting each image into each label as a separate folder for each
for img in tqdm(train.values):
    filename=img[0]
    label=img[1]
    #src
    src = os.path.join(data_root,'train_img',filename+'.png')
    label_dir = os.path.join(data_root,'train',label)
    dest = os.path.join(label_dir,filename+'.jpg')
    
    #img collection
    im = Image.open(src)
    rgb_im = im.convert('RGB')
    #image saving
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    rgb_im.save(dest)  
    
    if not os.path.exists(os.path.join(data_root,'train2',label)):
        os.makedirs(os.path.join(data_root,'train2',label))
    rgb_im.save(os.path.join(data_root,'train2',label,filename+'.jpg'))
    
#########################################################################
#-----------------------------------1.3---------------------------------#
#------------Prevention of overfitting(increase of accuracy)------------#
#########################################################################

#Randomizer for each image (flips, scales, blurs etc)
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest') #Change horizontal_flip to False when on robot

#src
src_train_dir = os.path.join(data_root,'train')
dest_train_dir = os.path.join(data_root,'train2')

#Counter
it = 0
#Number of items in folder you want
class_size = 600

for count in label_counts.values:
    #src
    dest_lab_dir = os.path.join(dest_train_dir,label_counts.index[it])
    src_lab_dir = os.path.join(src_train_dir,label_counts.index[it])
    
    if not os.path.exists(dest_lab_dir):
        os.makedirs(dest_lab_dir)
        
    #number of times to edit each image
    ratio = math.floor(class_size/count)-1
    print(count,count*(ratio+1),ratio)
        
    #Collection of image
    for file in os.listdir(src_lab_dir):
        img = load_img(os.path.join(src_lab_dir,file))
        img.save(os.path.join(dest_lab_dir,file))
        x = img_to_array(img) 
        x = x.reshape((1,) + x.shape)
        i = 0
        #creation of edited image, saved in train2
        for batch in datagen.flow(x, batch_size=1,save_to_dir=dest_lab_dir, save_format='jpg'):
            i += 1
            if i > ratio:
                break 
    it = it+1
'''  
#########################################################################
#------------------------------------2----------------------------------#
#------------------------Fine tuning the training-----------------------#
#########################################################################

#########################################################################
#-------------------------------Notes-----------------------------------#
#-------See for details. This section was purely text explaining--------#
#-------------how to manipulate code for desired response---------------#
#########################################################################


#########################################################################
#------------------------------------3----------------------------------#
#------------------------Testing on unseen records----------------------#
#########################################################################

#########################################################################
#-----------------------------------3.0---------------------------------#
#---------------------------------Imports-------------------------------#
#########################################################################
#The following 3 lines are required to be the first three lines of the program
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import pandas as pd
import argparse
import sys,os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

#########################################################################
#-----------------------------------3.1---------------------------------#
#---------------------------------Testing-------------------------------#
#########################################################################
t=tqdm(pd.read_csv('./Data/test.csv').values)
test=[]
i=0
for tt in t:
    test.append(tt[0])
    i+=1

def load_image(filename):
    #Read in the image_data to be classified."""
    return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
    #Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]

def load_graph(filename):
    #Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    
def run_graph(src, dest, labels, input_layer_name, output_layer_name, num_top_predictions):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        # predictions  will contain a two-dimensional array, where one
        # dimension represents the input image count, and the other has
        # predictions per class
        i=0
        #with open('submit.csv','w') as outfile:
        for f in os.listdir(src):
            if f != ".DS_Store":
                im=Image.open(os.path.join(src,f))
                img=im.convert('RGB')
                if not os.path.exists(dest):
                    os.makedirs(dest)
                img.save(os.path.join(dest,test[i]+'.jpg'))
                image_data=load_image(os.path.join(dest,test[i]+'.jpg'))
                softmax_tensor=sess.graph.get_tensor_by_name(output_layer_name)
                predictions,=sess.run(softmax_tensor, {input_layer_name: image_data})

                # Sort to show labels in order of confidence             
                top_k = predictions.argsort()[-num_top_predictions:][::-1]
                for node_id in top_k:
                    predicted_label = labels[node_id]
                    score = predictions[node_id]
                    print(test[i]+',',predicted_label)
                    #outfile.write(test[i]+','+human_string+'\n')
                i+=1

src = os.path.join('./Data/','MyImages')
dest = os.path.join('./Data/','test_img2')
labels = '/tmp/output_labels.txt'
graph = '/tmp/output_graph.pb'
input_layer = 'DecodeJpeg/contents:0'
output_layer = 'final_result:0'
num_top_predictions = 1
labels = load_labels(labels)
load_graph(graph)
run_graph(src,dest,labels,input_layer,output_layer,num_top_predictions)



