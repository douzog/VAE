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

E =2
L = 0.001

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=E, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

try:
    os.mkdir("net/")
    os.mkdir("cluster/")
except OSError:
    print ("net and cluster results exist" )
    os.system("rm -r net/")
    os.mkdir("net/")
    os.system("rm -r cluster/")
    os.mkdir("cluster/")

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(20, 64)
        self.hidden2 = nn.Linear(64, 16)
        self.hidden3 = nn.Linear(16, 4)

        # Output layer, 2 units - one for each digit
        self.output = nn.Linear(4, 2)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.hidden3(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
    
        return x

model = Network().to(device)
train_losses = []
val_losses = []
optimizer = torch.optim.Adam(model.parameters(), lr=L)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data , target= data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        np.save("net/output"+str(batch_idx)+".npy", output.detach().cpu().numpy())
        # print("data shape", data.shape)
        # print("target shape", target.shape)
        # print("output", output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_losses.append(train_loss / len(train_loader.dataset))
    np.save("net/NETWORKtrainloss.npy", train_losses)


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

def filename_return(label):
    f375grey = open("Test/coordinates/375coord_grey.txt", 'r')
    a375grey = stack_coord(f375grey, 375)
    f375grey.close()

    f400grey = open("Test/coordinates/400coord_grey.txt", 'r')
    a400grey = stack_coord(f400grey, 400)
    f400grey.close()

    f375white = open("Test/coordinates/375coord_white.txt", 'r')
    a375white = stack_coord(f375white, 375)
    f375white.close()

    f400white = open("Test/coordinates/400coord_white.txt", 'r')
    a400white = stack_coord(f400white, 400)
    f400white.close()

    print(label.shape)
    print("a375grey ", a375grey.shape)
    print("a400grey ", a400grey.shape)
    print("a375white ", a375white.shape)
    print("a400white ", a400white.shape)
    print("sum of all ",  a375grey.shape[0] +a400grey.shape[0] +a375white.shape[0] +  a400white.shape[0] )
    result = [0,0,0,0,0]


    # for i in range(0, len(label)):
    #     if label[i] == 0.:

    #         if a375grey[i] and a400grey[i] == True:
    #             result = np.vstack((result, a375grey[i]))
    #             result = np.vstack((result, a400grey[i]))
    #         else:
    #             result = np.vstack((result, a400grey[i]))
      
    #     elif label[i] == 1.:
    #         if a375white[i] and a400white[i] == True:
    #             result = np.vstack((result, a375white[i]))
    #             result = np.vstack((result, a400white[i]))
    #         else:
    #             result = np.vstack((result, a375white[i]))

    # for i in range(0, len(label)):

    #     if label[i] == 0.:

    #         if a375grey[i] and a400grey[i] == True:
    #             result = np.vstack((result, a375grey[i]))
    #             result = np.vstack((result, a400grey[i]))
    #         else:
    #             result = np.vstack((result, a400grey[i]))
      
    #     elif label[i] == 1.:
    #         if a375white[i] and a400white[i] == True:
    #             result = np.vstack((result, a375white[i]))
    #             result = np.vstack((result, a400white[i]))
    #         else:
    #             result = np.vstack((result, a375white[i]))

          

    print(result)
    result = np.delete(result, 0, axis=0)
    print("stacked results shape ", result.shape)
    print("label" , label.shape)
    return result

def load_output(size):
    result_array = np.load("net/output"+str(0)+".npy")
    for i in range(1,size):
        result = np.load("net/output"+str(i)+".npy")
        result_array = np.vstack((result_array, result))
    print(result_array.shape)
    return result_array


def kmeanss(Xt, X,y):

    #https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
    
    # np.random.seed(42)

    data = scale(Xt)
    n_samples, n_features = data.shape
    n_digits = len(np.unique(y))
    labels = y
    sample_size = 300


    print(data)
    print(data.shape)
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
        % (n_digits, n_samples, n_features))


    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')



def myKmeans(Xt, X,y, num_models):
    avg_score_array = np.array([])
    best_pred =  np.array([])
    np.random.shuffle(Xt)
    for i in range(0, num_models):
        # https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
        wcss = []
        Xt = PCA(n_components=2).fit_transform(Xt)

        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=20, random_state=0)
            kmeans.fit(Xt)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig("cluster/elbow"+str(i)+".png")
        plt.close()

        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=4, random_state=0)
        X = PCA(n_components=2).fit_transform(X)
        pred_y = kmeans.fit_predict(X)
        plt.scatter(X[:,0], X[:,1])
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.savefig("cluster/cluster"+str(i)+".png")
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
    tensor_test =  torch.from_numpy(mu_test)
    tensor_label_test = torch.from_numpy(label_test)
    test_dataset = data.TensorDataset(tensor_test, tensor_label_test)
    testset = data.DataLoader(test_dataset, batch_size=32)


    "SEND TO CLUSTERING, MAKE N MODELS, RETURN AVERAGE"
    pred_y, avg_score = myKmeans(Xv, mu_test, label_test, 10)
    print("truth     ", label_test)
    print("pred_y    " , pred_y)
    print("score avg " , avg_score)

    "WRITE OUT FILE FROM SAMPLE NUMBER, TRUE AND PREDICTED VALUE"
    for i in range(0, len(pred_y)):
        # print(str(pat/h[i])+ '\t' + str(label_test[i]) +'\t' + str(pred_y[i]))
        with open("cluster/cluster_results.txt", "a") as preds_file:
            preds_file.write(str(path[i])+ '\t' + str(label_test[i]) +'\t' + str(pred_y[i])  +'\n')

    print("( ° ͜ °)oO clustering done!")

    # final_result = coord()

    