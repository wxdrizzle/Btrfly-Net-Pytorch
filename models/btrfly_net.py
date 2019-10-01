import torch
import torch.nn as nn

def crop(input1, input2):
    assert input1.shape[0] == input2.shape[0]
    assert input1.shape[2] - input2.shape[2] in (0, 1)
    assert input1.shape[3] - input2.shape[3] in (0, 1)

    return (input1[:, :, :input2.shape[2], :input2.shape[3]], input2)

class conv_blk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, drop_out):
        super(conv_blk, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5 if drop_out else 0),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, input):
        output = self.blk(input)
        return output

class deconv_blk(nn.Module):
    def __init__(self, in_channels):
        super(deconv_blk, self).__init__()
        self.blk = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        output = self.blk(input)
        return output

class green(nn.Module):
    def __init__(self, cfg, pos):
        super(green, self).__init__()
        self.conv =conv_blk(in_channels=cfg.MODEL.CHANNELS[pos],
                            out_channels=cfg.MODEL.CHANNELS[pos+1],
                            kernel_size=1 if pos == 12 else 3,
                            stride=1,
                            padding=0 if pos == 12 else 1,
                            drop_out=True if pos in (4, 5) else False,
                            )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        output_side = self.conv(input)
        output_side_pad = nn.functional.pad(output_side, (0, (output_side.shape[3] % 2), 0, (output_side.shape[2] % 2), 0, 0, 0, 0))
        output_main = self.pool(output_side_pad)
        return output_main, output_side

class purple(nn.Module):
    def __init__(self, cfg, pos):
        super(purple, self).__init__()
        self.deconv = deconv_blk(in_channels=cfg.MODEL.CHANNELS[pos])
        self.conv = conv_blk(in_channels=cfg.MODEL.CHANNELS[pos] + (cfg.MODEL.CHANNELS[13 - pos] if pos < 9 else cfg.MODEL.CHANNELS[12 - pos]),
                             out_channels=cfg.MODEL.CHANNELS[pos+1],
                             kernel_size=3, stride=1, padding=1, drop_out=False,
                             )

    def forward(self, input_main, input_side):
        output = self.deconv(input_main)

        output = torch.cat(crop(output, input_side), dim=1)
        output = self.conv(output)
        return output

class red(nn.Module):
    def __init__(self, cfg, pos):
        super(red, self).__init__()
        self.blk = nn.Conv2d(
            in_channels=cfg.MODEL.CHANNELS[pos],
            out_channels=cfg.MODEL.CHANNELS[pos+1],
            kernel_size=1,stride=1, padding=0,
        )

    def forward(self, input):
        output = self.blk(input)
        return output



class in_arm(nn.Module):
    def __init__(self, cfg):
        super(in_arm, self).__init__()
        self.green0 = green(cfg, pos=0)
        self.green1 = green(cfg, pos=1)
        self.green2 = green(cfg, pos=2)

    def forward(self, input):
        output_main, output_side_0 = self.green0(input)
        output_main, output_side_1 = self.green1(output_main)
        output_main, output_side_2 = self.green2(output_main)
        return output_main, output_side_0, output_side_1, output_side_2


class out_arm(nn.Module):
    def __init__(self, cfg):
        super(out_arm, self).__init__()
        self.purple0 = purple(cfg, pos=9)
        self.purple1 = purple(cfg, pos=10)
        self.purple2 = purple(cfg, pos=11)
        self.red = red(cfg, pos=12)

    def forward(self, input_main, input_side_2, input_side_1, input_side_0):
        output = self.purple0(input_main, input_side_2)
        output = self.purple1(output, input_side_1)
        output = self.purple2(output, input_side_0)
        output = self.red(output)
        return output


class body(nn.Module):
    def __init__(self, cfg):
        super(body, self).__init__()
        self.green0 = green(cfg, pos=4)
        self.green1 = green(cfg, pos=5)
        self.conv = conv_blk(in_channels=cfg.MODEL.CHANNELS[6],
                             out_channels=cfg.MODEL.CHANNELS[7],
                             kernel_size=3, stride=1, padding=1, drop_out=True,
                             )
        self.purple0 = purple(cfg, pos=7)
        self.purple1 = purple(cfg, pos=8)

    def forward(self, input_sag, input_cor):
        output = torch.cat(crop(input_sag, input_cor), dim=1)
        output, side_0 = self.green0(output)
        output, side_1 = self.green1(output)
        output = self.conv(output)
        output = self.purple0(output, side_1)
        output = self.purple1(output, side_0)
        return output


class BtrflyNet(nn.Module):
    def __init__(self, cfg):
        super(BtrflyNet, self).__init__()
        self.input_arm_sag = in_arm(cfg)
        self.input_arm_cor = in_arm(cfg)
        self.body = body(cfg)
        self.output_arm_sag = out_arm(cfg)
        self.output_arm_cor = out_arm(cfg)

    def forward(self, sag, cor):
        sag_body, sag_side0, sag_side1, sag_side2 = self.input_arm_sag(sag)
        cor_body, cor_side0, cor_side1, cor_side2 = self.input_arm_cor(cor)
        body_out = self.body(sag_body, cor_body)
        out_sag = self.output_arm_sag(body_out, sag_side2, sag_side1, sag_side0)
        out_cor = self.output_arm_cor(body_out, cor_side2, cor_side1, cor_side0)
        return out_sag, out_cor