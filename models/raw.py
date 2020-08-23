import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, bottleneckFilters=32):
        super(Net, self).__init__()
        # 3 x 240 x 320
        self.con0 = nn.Conv2d(3, 8,  (7, 7),  padding=(3,3) ) 
        self.con1 = nn.Conv2d(8, 16, (5, 5),  padding=(2,2)) 
        self.con2 = nn.Conv2d(16, 32, (3, 3),  padding=(1,1)) 
        self.con3 = nn.Conv2d(32, 64, (3, 3),  padding=(1,1)) 
        self.con4 = nn.Conv2d(64, 64, (3, 3),  padding=(1,1)) 
        self.con5 = nn.Conv2d(64, bottleneckFilters,  (3, 3),  padding=(2,1)) 

        self.dcon0 = nn.Conv2d(bottleneckFilters, 16, (3, 3), padding=(1, 1)) 
        self.dcon1 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1)) 
        self.dcon2 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1)) 
        self.dcon3 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1)) 
        self.dcon4 = nn.Conv2d(128, 128, (3, 3), padding=(1, 1)) 
        self.dcon5 = nn.Conv2d(128, 3, (3, 3), padding=(1, 1))

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.tanh_a = nn.Tanh()

        self.encoded = None

    def forward(self, x):
        activations = self.encoder(x) # [-1, 1]
        # Stochastic binarization
        with torch.no_grad():
            rand = torch.rand(activations.shape).cuda()
            probs = (1 + activations) / 2
            eps = torch.zeros(activations.shape).cuda()
            eps[rand <= probs] = (1 - activations)[rand <= probs]
            eps[rand > probs] = (-activations - 1)[rand > probs]

        self.encoded = 0.5 * (activations + eps + 1)

        return self.decoder(self.encoded)

    def encoder(self, x):

        # 3 x 240 x 320
        x = self.pool(F.relu(self.con0(x)))
        # 8 x 120 x 160
        x = self.pool(F.relu(self.con1(x)))
        # 16 x 60 x 80
        x = self.pool(F.relu(self.con2(x)))
        # 32 x 30 x 40
        x = F.relu(self.con3(x))
        # 64 x 30 x 40
        x = self.pool(F.relu(self.con4(x)))
        # 64 x 15 x 20
        x = self.pool(self.con5(x))
        # 32 x 8 x 10
        return self.tanh_a(x)
      
    def decoder(self, encoded):
        y = encoded * 2.0 - 1

        # 32 x 8 x 10
        x0 = self.upsample(F.relu(self.dcon0(y)))
        x1 = self.upsample(F.relu(self.dcon1(x0)))
        x2 = self.upsample(F.relu(self.dcon2(x1)))
        x3 = self.upsample(F.relu(self.dcon3(x2)))
        x4 = self.upsample(F.relu(self.dcon4(x3)))
        x5 = F.relu(self.dcon5(x4))

        raw_out = x5[:, :, :240, :]

        return raw_out

if __name__ == "__main__":
    import numpy as np

    m = Net(bottleneckFilters=8)

    #array = torch.rand((1, 3, 240, 320)).cuda()

    #m(array)

    total = 0
    for param in m.parameters():
        layer_total = 1
        layer_dim = len(param.shape)
        for v in param.shape:
            layer_total *= v

        total += layer_total

    print(f"Wow, this model has {total} parameters.")
