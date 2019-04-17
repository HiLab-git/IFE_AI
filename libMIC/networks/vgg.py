import numpy as np
import torch
import torch.nn as nn

class ConvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
    """
    def __init__(self, in_channels, out_channels,
            kernel_size, stride = 1, padding = 0, dilation =1, groups = 1,
            bias = True, batch_norm = True, acti_func = None):
        super(ConvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        
        self.conv = nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias)
        if(self.batch_norm):
            self.bn = nn.modules.BatchNorm2d(out_channels)

    def forward(self, x):
        f = self.conv(x)
        if(self.batch_norm):
            f = self.bn(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f

class VGGBlock(nn.Module):
    def __init__(self,in_channels, out_channels,
            acti_func = None, conv_num = 2, pool = True):
        super(VGGBlock, self).__init__()
        self.conv_num = conv_num
        self.pool = pool
        
        assert(self.conv_num ==1 or self.conv_num ==2 or self.conv_num == 3)
        self.conv_layer1 = ConvolutionLayer(in_channels, out_channels,  3, 
                batch_norm = False, acti_func=acti_func)
        self.conv_layer2 = ConvolutionLayer(out_channels, out_channels, 3, 
                batch_norm = False, acti_func=acti_func)
        self.conv_layer3 = ConvolutionLayer(out_channels, out_channels, 3, 
                batch_norm = False, acti_func=acti_func)
        if(self.pool):
            self.pool_layer = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_layer1(x)
        if(self.conv_num  > 1):
            x = self.conv_layer2(x)
        if(self.conv_num  > 2):
            x = self.conv_layer3(x)
        if(self.pool):
            x = self.pool_layer(x)
        return x

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.block1 = VGGBlock(3, 16, 
             acti_func=self.get_acti_func(), pool = True, conv_num = 2)

        self.block2 = VGGBlock(16, 32,
             acti_func=self.get_acti_func(), pool = True, conv_num = 2)

        self.block3 = VGGBlock(32, 64,
             acti_func=self.get_acti_func(), pool = True, conv_num = 3)

        self.block4 = VGGBlock(64, 128,
             acti_func=self.get_acti_func(), pool = True, conv_num = 3)

        self.block5 = VGGBlock(128, 128,
             acti_func=self.get_acti_func(), pool = True, conv_num = 3)    

        self.drop = nn.Dropout2d(p = 0.5)

        self.fc1 = nn.Linear(128, 128)
        self.fc1_acti = self.get_acti_func()
        self.fc2 = nn.Linear(128, 64)
        self.fc2_acti = self.get_acti_func()
        self.fc3 = nn.Linear(64, 18)
        self.softmax = nn.Softmax(-1)
 
    def get_acti_func(self):
        return  nn.LeakyReLU(0.01)

    def forward(self, x):
        # input size 3x224x224
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x) # output size 128x1x1

        x = x.view(-1, 128)
        x = self.drop(x)
        x = self.fc1_acti(self.fc1(x))
        x = self.fc2_acti(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 9, 2)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    Net = VGGNet()
    Net = Net.double()

    x  = np.random.rand(2, 3, 224,224)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    y = y.detach().numpy()
    # print(y)
    print(y.shape)
