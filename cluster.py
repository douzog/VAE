from __future__ import print_function
import argparse
import torch
from torch import nn
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import torch.nn.init as init
import numpy as np
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.colors as colors
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Author: Isabella Douzoglou
# Partial Code: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

try:
    os.mkdir("cluster/")
except OSError:
    print ("cluster results exist" )
    os.system("rm -r cluster/")
    os.mkdir("cluster/")

def load2d(DATA, title, size):
    result_array = np.load(DATA+title+str(0)+".npy")
    for i in range(1,size):
        result = np.load(DATA+title+str(i)+".npy")
        result_array = np.vstack((result_array, result))
    return result_array

def load(DATA, title, size):
    result_array = np.array([])
    for i in range(0,size):
        result = np.load(DATA+title+str(i)+".npy")
        result_array = np.append(result_array, result)
    result_int = result_array.astype(int)
    return result_int

def stack_coord(filename, brain_num):
    result = [0,0,0,0,0]
    for line in filename.readlines():
        x = line.split()
        arr_ =[brain_num, int(x[1]), int(x[2]), int(x[3]), int(x[4])]
        # print(arr_)
        result = np.vstack((result, arr_))
    result = np.delete(result, 0, axis=0)
    return result

def load_output(size):
    result_array = np.load("net/output"+str(0)+".npy")
    for i in range(1,size):
        result = np.load("net/output"+str(i)+".npy")
        result_array = np.vstack((result_array, result))
    print(result_array.shape)
    return result_array

def myKmeans(Xt, X,y, num_models):
    avg_score_array = np.array([])
    np.random.shuffle(Xt)
    for i in range(0, num_models):
        # PCA init + KMEANS declare + fit + predict
        Xt = PCA(n_components=2).fit_transform(Xt)
        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=4, random_state=0)
        kmeans.fit(Xt)
        X = PCA(n_components=2).fit_transform(X)
        pred_y = kmeans.fit_predict(X)
        plt.scatter(X[:,0], X[:,1])
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.savefig("cluster/cluster"+str(i)+".png") # save plot
        plt.close()

        # check score if less than .3 then swap cluster
        score = accuracy_score(label_test, pred_y)
        if score <.3:
            pred_y = 1 - pred_y
        score = accuracy_score(label_test, pred_y)
        print(score)
        # append to best score array for average calculations
        avg_score_array = np.append(avg_score_array, score)
        # save prediction array with score
        np.save("cluster/"+str(score)+"_"+str(i)+"_y_pred.npy", pred_y)

    avg_score = np.average(avg_score_array)

    return pred_y, avg_score


if __name__ == "__main__":

    "EXPERIMENT NAME"
    TITLE= "BetaL1_"
    EPOCH =700
    N_WARM_UP = 100
    LEARNING_RATE = 0.0001 #1e-4 #(MINE)
    LOSSTYPE = "BETA"
    latent_dim = 20
    TEST = "eval_"+TITLE +"LR:"+str(LEARNING_RATE)+"_WU:"+str(N_WARM_UP)+"_E:"+str(1)+"_Ldim:"+str(latent_dim)+"/numpy/"
    TRAIN = "results_"+TITLE +"LR:"+str(LEARNING_RATE)+"_WU:"+str(N_WARM_UP)+"_E:"+str(EPOCH)+"_Ldim:"+str(latent_dim)+"/numpy/"
    print(TRAIN)
    print(TEST)

    "TRAIN SET" 
    Xt = load2d(TRAIN, "mu_train_", 4271)
    yt = load(TRAIN, "label_train_", 4271)

    "VAL SET" 
    Xv = load2d(TRAIN, "mu", 295)
    yv = load(TRAIN, "label", 295)
    tensor_train =  torch.from_numpy(Xt)
    tensor_label_train =  torch.from_numpy(yt)  
    train = data.TensorDataset(tensor_train, tensor_label_train)
    trainset = data.DataLoader(train, batch_size=32)
    tensor_cali =  torch.from_numpy(Xv)
    tensor_label_cali =  torch.from_numpy(yv)  
    cali_dataset = data.TensorDataset(tensor_cali, tensor_label_cali)
    caliset = data.DataLoader(cali_dataset, batch_size=32)

    "TESTING"
    mu_test = load2d(TEST, "mu", 231)
    label_test = load(TEST, "label", 231)
    path = load(TEST, "path", 231)
    tensor_test = torch.from_numpy(mu_test)
    tensor_label_test = torch.from_numpy(label_test)
    test_dataset = data.TensorDataset(tensor_test, tensor_label_test)
    testset = data.DataLoader(test_dataset, batch_size=32)

    "SEND TO CLUSTERING, MAKE N MODELS, RETURN AVERAGE"
    pred_y, avg_score = myKmeans(Xv, mu_test, label_test, 10) # mu performed better than sigma
    print("truth     ", label_test)
    print("pred_y    " , pred_y)
    print("score avg " , avg_score)

    "WRITE OUT FILE FROM SAMPLE NUMBER, TRUE AND PREDICTED VALUE"
    for i in range(0, len(pred_y)):
        with open("cluster/cluster_results.txt", "a") as preds_file:
            preds_file.write(str(path[i])+ '\t' + str(label_test[i]) +'\t' + str(pred_y[i])  +'\n')

    print("( ° ͜ °)oO clustering done!")

    
