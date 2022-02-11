
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convol1       = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1) 
        self.batch1      = nn.BatchNorm2d(num_features=128)
        self.transition1 = nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, padding=1)

        self.convol2       = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1) 
        self.batch2      = nn.BatchNorm2d(num_features=16)  
        self.pool1       = nn.MaxPool2d(2, 2)   
        

        self.convol3       = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.batch3      = nn.BatchNorm2d(num_features=16) 
        self.convol4       = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 
        self.batch4      = nn.BatchNorm2d(num_features=32)
        self.pool2       = nn.MaxPool2d(2, 2) 

        self.transition2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1)

        self.convol5       = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.batch5      = nn.BatchNorm2d(num_features=16) 
        self.convol6       = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 
        self.batch6      = nn.BatchNorm2d(num_features=32)

        self.convol7       = nn.Conv2d(in_channels=32, out_channels=25, kernel_size=1, padding=1)            
        self.gpool       = nn.AvgPool2d(kernel_size=7)
        self.drop        = nn.Dropout2d(0.1)


        

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

def test_func(test_images):
    test_images = test_images.reshape(test_images.shape[0], 1, 150, 150).astype('float32')
    test_data = []
    for i in range(len(test_images)):
        test_data.append(test_images[i]) 
    testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=64)
   
    model = torch.load('model.pt')
    model.eval()
    test_loss = 0
    correct = 0
    predicted_labels=[]
    with torch.no_grad():
        for data in testloader:
            gauss = transforms.GaussianBlur(7, sigma=(0.1, 2.0))
            data = gauss(data)
            data = F.interpolate(data, size=28)
            output = model(data)
            predicted_labels.append(output)
    predicted_labels = torch.argmax(torch.cat(predicted_labels,dim=0),dim=1).tolist()
    return predicted_labels


test_images = np.load('X_test.npy')  #test images can be inserted here
true_val = np.load('y_test.npy') #true val file can be inserted here
model = torch.load('model.pt')

predicted_labels = test_func(test_images)
correct_count=0
accuracy=0
for i in range(len(true_val)):      
    if(predicted_labels[i]==true_val[i]):
         correct_count += 1   
accuracy = (correct_count / len(true_val))*100  
print(accuracy)   
