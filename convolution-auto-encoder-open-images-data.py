# -*- coding: utf-8 -*-
"""

**Drive Mounting**
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive


from google.colab import drive
drive.mount('/content/gdrive') 


# %cd /content/gdrive/My Drive

"""**Libraries**"""

import csv
import pandas as pd
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""**Code**"""

def load_data():
    #from openimages.download import download_dataset
    #download_dataset("/content/gdrive/My Drive", ["Airplane"], annotation_format="pascal", limit=500)

    # perform transform on dataloaders
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

    batch_size = 32

    # dataset = torchvision.datasets.ImageFolder(root='./Images', transform=transform)
    dataset = torchvision.datasets.ImageFolder(root='./images', transform=transform)

    # calculate length of the data and split it in 80% train and 20% test
    lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
    subsetA, subsetB = torch.utils.data.random_split(dataset, lengths)

    # convert train and test data into dataloaders or batches of data
    train_loader= torch.utils.data.DataLoader(subsetA, batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(subsetB, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential( #784
                nn.Conv2d(3, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.Flatten(),
                nn.Linear(3136, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
                torch.nn.Linear(2, 3136),
                Reshape(-1, 64, 7, 7),
                nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, stride=(1, 1), kernel_size=(3, 3), padding=0),
                Trim(),  # 1x29x29 -> 1x28x28 (to shorten it otherwise 29x29 gives err)
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.model(x)
        return x

def train_autoencoder(train_loader, enc, dec):

    max_epochs = 500

    num_epochs = max_epochs
    learning_rate=1e-3

    batch_size = 32

    torch.manual_seed(42)
    criterion = F.mse_loss # mean square error loss
    enc_opt = torch.optim.Adam(enc.parameters(), lr=learning_rate) 
    dec_opt = torch.optim.Adam(dec.parameters(), lr=learning_rate)

    outputs = []
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            img, _ = data
            recon = dec(enc(img))
    
            loss = criterion(recon, img)
            loss.backward()
    
            enc_opt.step()
            dec_opt.step()

            enc_opt.zero_grad()
            dec_opt.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    torch.save(enc.state_dict(), "enc_model.pth")
    torch.save(dec.state_dict(), "dec_model.pth")

def load_autoencoder(test_loader):

    def to_img(x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        return x

    def show_image(img):
        img = to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def show_output(images, model_enc, model_dec):

        with torch.no_grad():

            images = images.to(device)
            images = model1(images)
            images = model2(images)
            images = images.cpu()
            images = to_img(images)
            np_imagegrid = torchvision.utils.make_grid(images[0:5], 5, 2).numpy()
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.show()

    images, labels = iter(test_loader).next()

    model_enc = Encoder()
    model_enc.load_state_dict(torch.load('enc_model.pth'))



    model_dec = Decoder()
    model_dec.load_state_dict(torch.load('dec_model.pth'))

    # First show original images
    print('Original images')
    show_image(torchvision.utils.make_grid(images[0:5],5,2))
    plt.show()

    # Reconstruct images using autoencoder
    print('Autoencoder images')
    show_output(images, model_enc, model_dec)
    
def main():
    train_loader, test_loader = load_data()
    enc = Encoder().to(device)
    dec = Decoder().to(device)
    train_autoencoder(train_loader, enc, dec)
    print('\n\n\n')
    load_autoencoder(test_loader)

if __name__ == '__main__':
    main()
