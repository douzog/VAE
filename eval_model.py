from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn.init as init
import numpy as np


TITLE= "BetaL1_"
EPOCH =1
N_WARM_UP = 100
LEARNING_RATE = 0.0001 #1e-4 #(MINE)
LOSSTYPE = "BETA"
latent_dim = 20
calc_shape = 256
pixel = 32
dim = pixel + pixel

NAME = TITLE +"LR:"+str(LEARNING_RATE)+"_WU:"+str(N_WARM_UP)+"_E:"+str(EPOCH)+"_Ldim:"+str(latent_dim)
print(NAME)


try:
    os.mkdir("eval_"+NAME+"/")
    os.mkdir("eval_"+NAME+"/numpy/")
except OSError:
    print ("eval folder exist, deleting re-creating" )
    os.system("rm -r eval_"+NAME+"/")
    os.mkdir("eval_"+NAME+"/")
    os.mkdir("eval_"+NAME+"/numpy/")

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=EPOCH, metavar='N',
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
no_cuda = True
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}



class MyData(datasets.ImageFolder):
    def __getitem__(self, idx):
        # this is what ImageFolder normally returns 
        original_tuple = super(MyData, self).__getitem__(idx)
        path = self.imgs[idx][0]
        img = np.array(original_tuple[0])
        label = np.array(original_tuple[1])
        # print("paht", path)

        path = path.strip('Testset/brain/white/')
        path = path.strip('Testset/brain/grey/')
        path = path.strip('.tif')
        path = int(path)

        # print("paht", path)
        # print(original_tuple)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        # print(tuple_with_path)
 
        return tuple_with_path
    def __len__(self):
        return len(self.imgs)

"LOAD TESTSET"
data_dir = "Testset/brain/"
dataset = MyData(data_dir, transform=transforms.ToTensor()) # our custom dataset
test_loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True, **kwargs)
test_losses = []


class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 256, 4, 1)

        self.fc11 = nn.Linear(calc_shape, latent_dim)
        self.fc12 = nn.Linear(calc_shape, latent_dim)

        self.fc2 = nn.Linear(latent_dim, 256)

        self.deconv1 = nn.ConvTranspose2d(256, 64, 4)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def encode(self, x):
        x = self.conv1(x)  # B,  32, 32, 32
        x = F.relu(x)
        x = self.conv2(x)  # B,  32, 16, 16
        x = F.relu(x)
        x = self.conv3(x)  # B,  64,  8,  8
        x = F.relu(x)
        x = self.conv4(x)  # B,  64,  4,  4
        x = F.relu(x)
        x = self.conv5(x)  # B, 256,  1,  1
        x = F.relu(x)

        flat = x.view((-1, 256 * 1 * 1))  # B, 256
        return self.fc11(flat), self.fc12(flat)

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc2(z)
        unflat = z.view(-1, 256, 1, 1)
        aggregated = unflat 
        x = F.relu(aggregated)
        x = self.deconv1(x)  # B,  64,  4,  4
        x = F.relu(x)
        x = self.deconv2(x)  # B,  64,  8,  8
        x = F.relu(x)
        x = F.relu(x)
        x = self.deconv3(x)  # B,  32, 16, 16
        x = F.relu(x)
        x = self.deconv4(x)  # B,  32, 32, 32
        x = F.relu(x)
        x = self.deconv5(x)  # B, nc, 64, 64
        return torch.sigmoid(x)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        decoded = self.decode(z)
        return decoded, mu, log_sigma

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class DeterministicWarmup(object):
    def __init__(self, n_steps, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.increase = self.t_max / n_steps
    def __iter__(self):
        return self
    def __next__(self):
        t = self.t + self.increase
        self.t = self.t_max if t > self.t_max else t
        return self.t
        

def test(epoch):
    model.eval()
    test_loss = 0
    num_batches = 0
    with torch.no_grad():
        for i, (data, label, path) in enumerate(test_loader):
 
            # print(data)
            # print(label.shape)
            # print(path.shape)
            # label = label.to(device)
            # path = path.to(device)
            im = data.to(device)
            recon_batch, mu, logvar = model(im)
            np.save("eval_"+NAME+"/numpy/mu"+ str(num_batches)+ ".npy", mu.detach().cpu().numpy())
            np.save("eval_"+NAME+"/numpy/sigma"+ str(num_batches)+ ".npy", logvar.detach().cpu().numpy())
            np.save("eval_"+NAME+"/numpy/label"+ str(num_batches)+ ".npy", label.detach().cpu().numpy())
            np.save("eval_"+NAME+"/numpy/path"+ str(num_batches)+ ".npy", path.detach().cpu().numpy())

            munp = mu.detach().cpu().numpy()
            lognp = logvar.detach().cpu().numpy()
            labelnp = label.detach().cpu().numpy()

            "LOSS L1"
            
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            likelihood = - F.l1_loss(recon_batch, data, reduction='sum')
            elbo = likelihood - torch.sum(KLD)
            loss = - elbo / len(data)
            test_loss += loss.item()
            num_batches += 1

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 3, 64, 64)[:n]])
                save_image(comparison.cpu(),
                         'eval_'+NAME+'/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    if epoch >0:
        test_losses.append(test_loss)
    np.save(NAME+"_eval_loss.npy", test_losses)


if __name__ == "__main__":
    "GET MODEL"
    model = BetaVAE().to(device)
    model.load_state_dict(torch.load("BetaL1_LR:0.0001_WU:100_E:700_Ldim:20.pt"))

    model.eval()

    result_label = np.array([])
    for epoch in range(1, args.epochs + 1):
        test(epoch)
