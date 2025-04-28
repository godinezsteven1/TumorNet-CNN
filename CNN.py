import torch
import torch.nn as nn
import torch.nn.functional as F


# lets make a basic CNN neural network as of now and change it based on needs 

# MRI scans are in black and white so we can use a gray scale channel instead of 3 (rgb)
# --------------Convolution parameters ---------------------------
grayscaleChannel = 1 # -  image input channel -- 1 input for grayscale 
out_channels = 64 # this isnt a MNSIT + Need complex features to be learned 
kernel_size = 3 # cannot go wrong w 3 
imagePadding = 1 # works with kernel size 3 
stride = 2 # idea for downsampling with out to much overlap 
# --------------Neural Network Parameters ------------------------
inputFeature = 256 * 6 * 6 # approx 9k features 
outputClasses = 4 
neuronsLayer1 = 1024
neuronsLayer2 = 256
p = 0.2
# 9216 --> 1024 --> 256 --> 4 
# -----------------------------------------------------------------


# nn.module - base class for all neural networks -- model needs to subclass 
class CNN_Net(nn.Module):
    def __init__(self): 
        super().__init__() # submodule 

        # lets start with 3 layers of convolutions 
        # maybe 3 fully connected layers 
        
        self.conv1Layer = nn.Conv2d(
            in_channels=grayscaleChannel, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=imagePadding)

        self.conv2Layer = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels*2, # double at each layer
            kernel_size=kernel_size,
            padding=imagePadding)

        self.conv3Layer = nn.Conv2d(
            in_channels=out_channels*2, 
            out_channels=out_channels*4, 
            kernel_size=kernel_size,
            padding=imagePadding)
        # max pool kernel -- size and stride 
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        #in features, out features 
        # 63 * 63 * 256 # chat made these calculations however 1M + input feature is crazy 
        # compress  down to 6×6,  without choosing a fixed kernel or stride manually
        # so now well update input in line 14
        self.adaptivePool = nn.AdaptiveAvgPool2d((6, 6))
        self.hiddenLayer1 = nn.Linear(inputFeature, neuronsLayer1)
        self.hiddenLayer2 = nn.Linear(neuronsLayer1, neuronsLayer2)
        self.output = nn.Linear(neuronsLayer2, outputClasses)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.conv1Layer(x)
        # because it is a 4 class clasification going to relu to keep x values 
        x = F.relu(x)
        x = self.pool(x) # reduce dim

        x = self.conv2Layer(x)
        x = F.relu(x) 
        x = self.pool(x)

        x = self.conv3Layer(x)
        x = F.relu(x)
        x = self.pool(x) 

        # applied 3 convolutional layers -- now apply connected layers 
        # flatten features 
        x = x.view(-1, inputFeature)

        x = self.hiddenLayer1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.hiddenLayer2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.output(x)

        return x 
        

# still need to apply a cross entoorpu loss or a  softmax maybe in the visuals 
# cel easier i beleive pytorch version applies a softmax at the end. 
