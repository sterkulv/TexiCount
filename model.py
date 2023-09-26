import torch
import torch.nn as nn
import torch.nn.init as init

class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, pad=0, activation=nn.LeakyReLU()):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv1(x)))


class SimpleBlock(nn.Module):
    def __init__(self, in_chan, out_chan_1x1, out_chan_3x3, activation=nn.LeakyReLU(), dropout_prob=0.2):
        super(SimpleBlock, self).__init__()
        self.conv1 = ConvBlock(in_chan, out_chan_1x1, ksize=1, pad=0, activation=activation)
        self.conv2 = ConvBlock(in_chan, out_chan_3x3, ksize=3, pad=1, activation=activation)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        output = torch.cat([conv1_out, conv2_out], 1)
        return output


class ModelCountception(nn.Module):
    def __init__(self, width, height, dropout, inplanes=3, outplanes=1):
        super(ModelCountception, self).__init__()
        # params
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.01)
        self.padding_size = 32
        self.dropout = dropout
        torch.LongTensor()
        self.feature_maps = {}

        self.conv1 = ConvBlock(self.inplanes, 64, ksize=3, pad=self.padding_size, activation=self.activation)
        self.simple1 = SimpleBlock(64, 16, 16, activation=self.activation)
        self.simple2 = SimpleBlock(32, 16, 32, activation=self.activation)
        self.conv2 = ConvBlock(48, 16, ksize=3, activation=self.activation)
        #self.simple3 = SimpleBlock(16, 112, 48, activation=self.activation)
        #self.simple4 = SimpleBlock(160, 32, 96, activation=self.activation)
        #self.conv3 = ConvBlock(128, 32, ksize=5, activation=self.activation)
        #self.conv4 = ConvBlock(32, self.outplanes, ksize=1, activation=self.final_activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1.register_forward_hook(self.save_feature_map('conv1'))
        self.simple1.register_forward_hook(self.save_feature_map('simple1'))
        self.simple2.register_forward_hook(self.save_feature_map('simple2'))
        self.conv2.register_forward_hook(self.save_feature_map('conv2'))
        #self.simple3.register_forward_hook(self.save_feature_map('simple3'))
        #self.simple4.register_forward_hook(self.save_feature_map('simple4'))
        #self.conv3.register_forward_hook(self.save_feature_map('conv3'))
        #self.conv4.register_forward_hook(self.save_feature_map('conv4'))
        self.pool.register_forward_hook(self.save_feature_map('pool'))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(408960, 64) #12075 for 200x180. 24920 for 225x300
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(64, self.outplanes)
        self.softmax = nn.Softmax(dim=1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def save_feature_map(self, layer_name):
        def hook(module, input, output):
            #self.feature_map = output.detach()
            self.feature_maps[layer_name] = output.detach()
        return hook
    
    def forward(self, x):
        net = self.conv1(x)
        net = self.simple1(net)
        net = self.simple2(net)
        net = self.conv2(net)
        #net = self.simple3(net)
        #net = self.simple4(net)
        #net = self.conv3(net)
        #net = self.conv4(net)
        net = self.pool(net)

        x = self.flatten(net)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

def create_model(width, height, dropout, outplanes):
    model = ModelCountception(width, height, dropout, outplanes=outplanes)
    return model


class ModelTexiCount(nn.Module):
    def __init__(self, width, height, dropout, inplanes=3, outplanes=1):
        super(ModelTexiCount, self).__init__()
        # params
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.01)
        self.dropout = dropout
        torch.LongTensor()
        self.feature_maps = {}
        
        self.conv1 = nn.Conv2d(self.inplanes, 64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=3, padding=1) #Maybe try to change this
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 64, kernel_size=5, padding=2)
        self.batch_norm3 = nn.BatchNorm2d(128)
        #self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        #self.batch_norm4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, self.outplanes, kernel_size=1, padding=0)
        #self.batch_norm5 = nn.BatchNorm2d(self.outplanes)
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1_activation = self.activation
        self.conv2_activation = self.activation
        self.conv3_activation = self.activation
        #self.conv4_activation = self.activation
        self.conv4_activation = self.final_activation
        

        self.conv1.register_forward_hook(self.save_feature_map('conv1'))
        self.conv2.register_forward_hook(self.save_feature_map('conv2'))
        self.conv3.register_forward_hook(self.save_feature_map('conv3'))
        self.conv4.register_forward_hook(self.save_feature_map('conv4'))
        #self.conv5.register_forward_hook(self.save_feature_map('conv5'))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(width*height*self.outplanes, 32)
        #self.dropout_layer = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, self.outplanes)
        self.softmax = nn.Softmax(dim=1)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def save_feature_map(self, layer_name):
        def hook(module, input, output):
            #self.feature_map = output.detach()
            self.feature_maps[layer_name] = output.detach()
        return hook



    def forward(self, x):
        x = self.conv1_activation(self.batch_norm1(self.conv1(x)))
        x = self.conv2_activation(self.batch_norm2(self.conv2(x)))
        x = self.conv3_activation(self.batch_norm3(self.conv3(x)))
        #x = self.conv4_activation(self.batch_norm4(self.conv4(x)))
        x = self.conv4_activation(self.conv4(x))
        #x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.relu(x)

        return x
    
def create_TexiCount(width, height, dropout, outplanes):
    model = ModelTexiCount(width, height, dropout, outplanes=outplanes)
    return model

class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, pad=0, activation=nn.LeakyReLU()):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv1(x)))


class ModelTexiCount_old(nn.Module):
    def __init__(self, width, height, dropout, inplanes=3, outplanes=1):
        super(ModelTexiCount_old, self).__init__()
        # params
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.01)
        self.dropout = dropout
        self.padding_size = 32
        self.feature_maps = {}
        torch.LongTensor()

        self.conv1 = ConvBlock(self.inplanes, 64, ksize=3, activation=self.activation)
        self.conv2 = ConvBlock(64, 96, ksize=3, activation=self.activation)
        self.conv3 = ConvBlock(96, 64, ksize=5, activation=self.activation)
        self.conv4 = ConvBlock(64, self.outplanes, ksize=1, activation=self.final_activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Saving feature maps to visualize output from convnet
        self.conv1.register_forward_hook(self.save_feature_map('conv1'))
        self.conv2.register_forward_hook(self.save_feature_map('conv2'))
        self.conv3.register_forward_hook(self.save_feature_map('conv3'))
        self.conv4.register_forward_hook(self.save_feature_map('conv4'))
        self.pool.register_forward_hook(self.save_feature_map('pool'))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(round((width-8)/2))*int(((height-8)/2)), 64) #12075 for 200x180. 24920 for 225x300
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(64, self.outplanes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def save_feature_map(self, layer_name):
        def hook(module, input, output):
            #self.feature_map = output.detach()
            self.feature_maps[layer_name] = output.detach()
        return hook


    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.pool(net)

        x = self.flatten(net)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.relu(x)

        return x
    
def create_TexiCount_old(width, height, dropout, outplanes):
    model = ModelTexiCount_old(width, height, dropout, outplanes=outplanes)
    return model