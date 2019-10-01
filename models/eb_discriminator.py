import torch
import torch.nn as nn

class EBGAN(nn.Module):
    """

    """
    def __init__(self):
        super(EBGAN, self).__init__()

        self.avgpool3d = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.architecture = nn.Sequential(
            #
            nn.Conv3d(in_channels=1, out_channels=5, kernel_size=(5, 5, 5), padding=(2, 2, 2), stride=(1, 1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(5),
            #
            nn.Conv3d(in_channels=5, out_channels=10, kernel_size=(5, 5, 5), padding=(2, 2, 2), stride=(1, 1, 1)),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(10),
            #
            nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(5, 5, 5), padding=(2, 4, 4), dilation=(1, 2, 2)),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(10),
            #
            nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(5, 5, 5), padding=(2, 4, 4), dilation=(1, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(10),
            #
            nn.ConvTranspose3d(in_channels=10, out_channels=5, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(5),
            #
            nn.ConvTranspose3d(in_channels=5, out_channels=5, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(5),
            #
            nn.Conv3d(in_channels=5, out_channels=1, kernel_size=(1,1,1))

        )


    def forward(self, input):
        input2 = input[:, 1:25, :, :].view(input.shape[0], 1, 24, input.shape[2], input.shape[3])
        reduce_input = self.avgpool3d(input2)
        output = self.architecture(reduce_input)
        D = pow((reduce_input - output),2)
        #D.view(-1)#D.sum(-1)
        return D.view(-1).sum(-1)