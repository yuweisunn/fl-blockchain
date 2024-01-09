import numpy as np 
import cv2 
from cnn.ConvNet import *
import random
from cnn.common.optimizer import RMSProp
from module.builder import *



def train(node_name, local_params_init,  data, label):  
    
    print(data.shape)
    train_accuracy = []
    
    epochs = 10
    batch_size = 32
    xsize = data.shape[0]
    iter_num = np.ceil(xsize / batch_size).astype(np.int)
    
    snet =ConvNet(input_dim=(1, 28, 28), 
                 conv_param={'filter1_num':5, 'filter1_size':3, 'filter2_num':5, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 2 ,use_dropout = False, dropout_ration = 0.1, use_batchnorm = False)
    
    snet.params = local_params_init
    snet.layer()
    optimizer = RMSProp(lr=0.00001, rho=0.9)
    
    epoch_list = []

    for epoch in range(epochs):
        epoch_list.append("epoch %s" %(epoch+1))

        idx = np.arange(xsize)
        np.random.shuffle(idx)
    
        for it in range(iter_num):
            mask = idx[batch_size*it : batch_size*(it+1)]
            # ミニバッチの生成
            x_train = data[mask]
            t_train = label[mask]
            
            # 勾配の計算 (誤差逆伝播法を用いる) 
            grads = snet.gradient(x_train, t_train)
            optimizer.update(snet.params, grads)
    
        train_accuracy.append(snet.accuracy(data, label ,batch_size))
        
        if (epoch+1)%1 == 0:
            print("Retraining the local model: %s/%s" %((epoch+1), epochs))#, train_accuracy[-1])
            
        
    model_name = "./update/%s.npy" %node_name

    np.save(model_name, snet.params) 
    
