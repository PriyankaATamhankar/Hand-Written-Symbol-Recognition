import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# class defining the neural network and layers within the network and functions for initialization and forward path
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convol1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(num_features=128)
        self.transition1 = nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, padding=1)

        self.convol2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convol3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(num_features=16)
        self.convol4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.transition2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1)

        self.convol5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(num_features=16)
        self.convol6 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batch6 = nn.BatchNorm2d(num_features=32)

        self.convol7 = nn.Conv2d(in_channels=32, out_channels=25, kernel_size=1, padding=1)
        self.gpool = nn.AvgPool2d(kernel_size=7)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, out):
        out = F.relu(self.convol1(out))
        out = self.batch1(out)
        out = self.drop(out)
        out = self.transition1(out)
        out = F.relu(self.convol2(out))
        out = self.batch2(out)
        out = self.drop(out)
        out = self.pool1(out)

        out = F.relu(self.convol3(out))
        out = self.batch3(out)
        out = self.drop(out)
        out = F.relu(self.convol4(out))
        out = self.batch4(out)
        out = self.drop(out)
        out = self.pool2(out)
        out = self.transition2(out)

        out = F.relu(self.convol5(out))
        out = self.batch5(out)
        out = self.drop(out)
        out = F.relu(self.convol6(out))
        out = self.batch6(out)
        out = self.drop(out)

        out = self.convol7(out)
        out = self.gpool(out)
        out = out.view(-1, 25)
        return F.log_softmax(out)

#function for training the model
def training(model, trainloader, optimizer, epoch):
    model.train()
    progressbar = tqdm(trainloader)
    for batch_idx, (Images, Labels) in enumerate(progressbar):
        gauss = transforms.GaussianBlur(7, sigma=(0.1, 2.0))
        Images = gauss(Images)
        Images = F.interpolate(Images, size=28)
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(Images)
        Labels = Labels.type(torch.LongTensor)
        predicted_labels = Labels
        loss = F.nll_loss(output, Labels)
        loss.backward()
        optimizer.step()
        progressbar.set_description(desc=f'epoch: {epoch} loss={loss.item()} batch_id={batch_idx}')

#Evaluating accuracy on validation data
def val(model, valloader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for Images, Labels in valloader:
            gauss = transforms.GaussianBlur(7, sigma=(0.1, 2.0))
            Images = gauss(Images)
            Images = F.interpolate(Images, size=28)
            output = model(Images)
            Labels = Labels.type(torch.LongTensor)
            loss = F.nll_loss(output, Labels)
            val_loss += F.nll_loss(output, Labels, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(Labels.view_as(pred)).sum().item()

    val_loss /= len(valloader.dataset)
    accuracy = 100. * correct / len(valloader.dataset)
    print(val_loss, correct, len(valloader.dataset), accuracy)
    return accuracy
	
	
#main function
#loading data from .npy files
X_train = np.load('X_train.npy')
y_train = np.load('y_Train.npy')
#splitting the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=10)

# validation= 10% of data set
X_train = X_train.reshape((X_train.shape[0], 1, 150, 150)).astype('float32')
X_val = X_val.reshape(X_val.shape[0], 1, 150, 150).astype('float32')

#loading data in trainloader for image processing before passing the data to the neural network
train_data = []
for i in range(len(X_train)):
    train_data.append([X_train[i], y_train[i]])
trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=64)

#loading data in valloader for image processing before passing the data to the neural network
val_data = []
for i in range(len(X_val)):
    val_data.append([X_val[i], y_val[i]])
    valloader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=64)

#passing data through the neural network and setting limit on number of epochs to avoid overfitting
model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
acc = 0
for epoch in range(1, 120):
    acc = 0
    training(model, trainloader, optimizer, epoch)
    acc = val(model, valloader)
    if (acc >= 96):
        break
print(epoch)

#saving the trained model in local directory
torch.save(model, 'model.pt')
