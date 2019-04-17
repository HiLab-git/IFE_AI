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
            bias = True, batch_norm = True, acti_func = None, keep_prob = 1.0):
        super(ConvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        self.keep_prob  = keep_prob
        
        self.conv = nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias)
        if(self.batch_norm):
            self.bn = nn.modules.BatchNorm2d(out_channels)
        if(self.keep_prob < 1.0):
            self.drop = nn.Dropout2d(p = 1.0 - self.keep_prob)

    def forward(self, x):
        f = self.conv(x)
        if(self.batch_norm):
            f = self.bn(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        if(self.keep_prob < 1.0):
            f = self.drop(f)
        return f

class VGGBlock(nn.Module):
    def __init__(self,in_channels, out_channels,
            kernel_size, padding = 0, acti_func = None, keep_prob = 1.0,
            pool = True, conv_num = 2):
        super(VGGBlock, self).__init__()

        self.pool = pool
        self.conv_num = conv_num
        assert(self.conv_num ==2 or self.conv_num == 3)
        self.conv_layer1 = ConvolutionLayer(in_channels, out_channels, kernel_size, 
                padding = padding, acti_func=acti_func, keep_prob=keep_prob)
        self.conv_layer2 = ConvolutionLayer(out_channels, out_channels, kernel_size, 
                padding = padding, acti_func=acti_func, keep_prob=keep_prob)
        self.conv_layer3 = ConvolutionLayer(out_channels, out_channels, kernel_size, 
                padding = padding, acti_func=acti_func, keep_prob=keep_prob)
        if(self.pool):
            self.pool_layer = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        if(self.conv_num == 3):
            x = self.conv_layer3(x)
        if(self.pool):
            x = self.pool_layer(x)
        return x

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.block1 = VGGBlock(3, 16, 3, padding = 1,
             acti_func=self.get_acti_func(), keep_prob = 1.0,
             pool = True, conv_num = 2)

        self.block2 = VGGBlock(16, 32, 3, padding = 1,
             acti_func=self.get_acti_func(), keep_prob = 1.0,
             pool = True, conv_num = 2)

        self.block3 = VGGBlock(32, 64, 3, padding = 1,
             acti_func=self.get_acti_func(), keep_prob = 1.0,
             pool = True, conv_num = 3)

        self.block4 = VGGBlock(64, 128, 3, padding = 1,
             acti_func=self.get_acti_func(), keep_prob = 0.5,
             pool = True, conv_num = 3)

        self.block5 = VGGBlock(128, 128, 3, padding = 1,
             acti_func=self.get_acti_func(), keep_prob = 0.5,
             pool = True, conv_num = 3)    
        
        self.block6 = VGGBlock(128, 32, 3, padding = 0,
             acti_func=self.get_acti_func(), keep_prob = 1.0,
             pool = False, conv_num = 2)  

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc1_acti = self.get_acti_func()
        self.fc2 = nn.Linear(128, 64)
        self.fc2_acti = self.get_acti_func()
        self.fc3 = nn.Linear(64, 18)
        self.softmax = nn.Softmax(-1)
    def get_acti_func(self):
        return  nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view(-1, 32*4*4)
        x = self.fc1_acti(self.fc1(x))
        x = self.fc2_acti(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 9, 2)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    Net = VGGNet()
    Net = Net.double()

    x  = np.random.rand(2, 3, 256,256)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    y = y.detach().numpy()
    print(y)
    print(y.shape)
