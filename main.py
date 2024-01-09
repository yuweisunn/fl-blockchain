import numpy as np 
import cv2 
import pandas as pd
import random
from cnn.ConvNet import *
from cnn.common.optimizer import RMSProp
from module.train import *
import pickle
import matplotlib.pyplot as plt
import csv
from BCModel import *
from creatBlock import *
from keras.datasets import mnist
import keras
from numpy.random import rand

th_list = []
quality_list = {0:[],1:[],2:[]}

blockchain = Blockchain()
blockchain.create_genesis_block()  
peers = set()
    
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32').reshape(-1,1,28,28)  
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.astype('float32').reshape(-1,1,28,28)[:1000] 
y_test = keras.utils.to_categorical(y_test, 10)[:1000] 

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def occlusion(image,prob):
    w = image.shape[0]
    output = image[0]
    output = cv2.circle(output, (7+int(rand()*w/2), 7+int(rand()*w/2)), int(prob/2),(0, 0, 0), -1)
    
    return output


def download():
    global_params = np.load("global.npy", allow_pickle = True).item()

    return global_params 
    
def upload(node_name):  
    local_params = np.load("update/%s.npy" %node_name, allow_pickle = True).item() 

    return local_params

def chain(blockchain):
    addNewUpdate(blockchain)
    print("added to buffer")
    result = blockchain.mine()
    if not result:
        print("No transactions to mine")
    else:
        # Making sure we have the longest chain before announcing to the network
        chain_length = len(blockchain.chain)
        #consensus()
        if chain_length == len(blockchain.chain):
            # announce the recently mined block to the network
            announce_new_block(blockchain.last_block,peers)
        print( "Block #{} is mined.".format(blockchain.last_block.index))
        print('------------------------------')
        print(blockchain.chain[-1].hash)
        print('------------------------------')
    

def Execute(node_name, data, label, global_params):
    local_params_init = download()
    train(node_name, local_params_init, data, label)
    local_params = upload(node_name)
    
    quality = precision(local_params, x_test, y_test) 
    threshold = precision(global_params, x_test, y_test)- 0.03
    if node_name == 0:
        th_list.append(threshold)
        
    quality_list[node_name].append(quality)
    
    print("=============================")
    print(quality)
    print(threshold) 
    print("=============================")
    
    
    if quality < threshold:
        return []
    
    chain(blockchain)
    
    return local_params


def Aggregate(local_parameters_list, global_params):
    for k in local_parameters_list[0].keys():
         temp = []   
         
         for i in range(len(local_parameters_list)):
             temp.append(local_parameters_list[i][k])   
             global_params[k] = np.mean(temp, axis=0)
             
    model_name = "global"
    np.save(model_name, global_params)    
    
    
def precision(local_params, train_data, label):
    label = np.array(label)
    
    snet = ConvNet(input_dim=(1, 28, 28), 
                 conv_param={'filter1_num':1, 'filter1_size':3, 'filter2_num':1, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 10 ,use_dropout = False, dropout_ration = 0.1, use_batchnorm = False)
    
    snet.params = local_params
    snet.layer()
    acc = snet.accuracy(train_data, label, 125)
    
    return acc

def main(num, noise_lvl):
    max_rounds = 20
    
    global_params = {}
    print("Systems initialize...")

    osNet =ConvNet(input_dim=(1, 28, 28),
                 conv_param={'filter1_num':1, 'filter1_size':3, 'filter2_num':1, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 10 ,use_dropout = False, dropout_ration = 0.1, use_batchnorm = False)
    
    global_params = osNet.params 
    osNet.layer()
    
    model_name = 'global'
    np.save(model_name, osNet.params) 

       
    
    acc_list = {}
    
    for i in range(max_rounds):
        local_parameters_list = []
        acc_list[i] = []
        for j in range(num): 
            print("[Round:%s][%s] Downloading global parameters and retraining in the local..." % (i+1, j))
            dex_left = i*(60000//max_rounds)+j*(60000//(num*max_rounds))
            index_right = i*(60000//max_rounds)+(j+1)*(60000//(num*max_rounds))
            data = x_train[index_left:index_right] 
            label = y_train[index_left:index_right]
            
            global_params = download()
            acc = precision(global_params, data, label)
            acc_list[i].append(acc)
            print("[Round:%s][%s] ML detection precision: %s" %(i+1, j, acc))
            
            #add noise
            if j == 0 and noise_lvl != 0:
                for h in range(len(data)):
                    data[h] = occlusion(data[h],noise_lvl)  
            local_params = Execute(j, data, label, global_params)
            
            if local_params != []:
                
                local_parameters_list.append(local_params)
                print("[Round:%s][%s] Uploading new local parameters..." % (i+1, j))
            
            os.remove("update/%s.npy" %j)
            
        Aggregate(local_parameters_list, global_params)
        print("[Round:%s] Aggregating successfully" % (i+1))  
        
        
    with open('BC.pkl', 'wb') as output:
        pickle.dump(blockchain, output)    
    print(acc_list)
    print('+++++++++++++++++++++++++++++')
    print(th_list)
    print('+++++++++++++++++++++++++++++')
    print(quality_list[0])
    print('+++++++++++++++++++++++++++++')
    print(quality_list[1])
    print('+++++++++++++++++++++++++++++')
    print(quality_list[2])
    
    
main(3,0)

