import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, dims, drop=0.2, half=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dims[0], dims[0], kernel_size=5, padding='same', groups=dims[0])  # depthwise conv
        self.pwconv = nn.Conv2d(dims[0], dims[1], kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm2d(dims[0])
        self.bn2 = nn.BatchNorm2d(dims[1])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if not half else nn.MaxPool2d(kernel_size=(1, 2),
                                                                                           stride=(1, 2))

    def forward(self, x):
        x = self.dwconv(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pwconv(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        return x


class CMFRM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.ch_linear1 = nn.Linear(dim*4, 512)
        self.ch_linear2 = nn.Linear(512,dim*2)
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv2d(dim*2, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1)
        self.relu = nn.ReLU()

    def channel_wise(self, input_rgb, input_depth):
        print(input_rgb.shape)
        print(input_depth.shape)
        rgb_max = F.max_pool2d(input_rgb, kernel_size=input_rgb.size()[2:])
        rgb_avg = F.avg_pool2d(input_rgb, kernel_size=input_rgb.size()[2:])
        depth_max = F.max_pool2d(input_depth, kernel_size=input_depth.size()[2:])
        depth_avg = F.avg_pool2d(input_depth, kernel_size=input_depth.size()[2:])

        x = torch.cat([rgb_max, rgb_avg, depth_max, depth_avg], dim=1)
        x = torch.squeeze(x)
        x = self.ch_linear1(x)
        x = self.sigmoid(x)
        x = self.ch_linear2(x)
        
        rgb, depth = torch.tensor_split(x, 2, dim=1)
        rgb = torch.mul(rgb.unsqueeze(dim=-1).unsqueeze(dim=-1), input_rgb)
        depth = torch.mul(depth.unsqueeze(dim=-1).unsqueeze(dim=-1), input_depth)

        return rgb, depth

    def spatial_wise(self, input_rgb, input_depth):
        x = torch.cat([input_rgb, input_depth], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        rgb, depth = torch.tensor_split(x, 2, dim=1)

        rgb = torch.mul(rgb, input_rgb)
        depth = torch.mul(depth, input_depth)
        return rgb, depth

    def forward(self, tuple):
        input_rgb = tuple[0]
        input_depth = tuple[1]

        channel_rgb, channel_depth = self.channel_wise(input_rgb, input_depth)
        spatial_rgb, spatial_depth = self.spatial_wise(input_rgb, input_depth)

        rect_rgb = torch.add(0.5 * channel_rgb, 0.5 * spatial_rgb)
        rect_depth = torch.add(0.5 * channel_depth, 0.5 * spatial_depth)

        rgb = torch.add(input_rgb, rect_rgb)
        depth = torch.add(input_depth, rect_depth)

        return rgb, depth


class Net_RGBD(nn.Module):
    def __init__(self, dims=[64, 128, 256, 256]):
        super(Net_RGBD, self).__init__()

        self.stem_rgb = nn.Conv2d(3, dims[0], kernel_size=2, stride=2)
        self.stem_depth = nn.Conv2d(1, dims[0], kernel_size=2, stride=2)

        self.depth = len(dims) - 1
        self.blocks_rgb = nn.ModuleList()
        self.blocks_depth = nn.ModuleList()
        self.cmfrms = nn.ModuleList()
        for k in range(self.depth):
            self.blocks_rgb.append(Block(dims=[dims[k], dims[k + 1]]))
            self.blocks_depth.append(Block(dims=[dims[k], dims[k + 1]]))
            self.cmfrms.append(CMFRM(dims[k+1]))

        self.dense1 = nn.Linear(16384, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(128, 1, bias=True)

        self.relu = torch.nn.ReLU()

    def forward(self, rgb, depth):
        rgb = self.stem_rgb(rgb)
        depth = self.stem_depth(depth)

        for i in range(self.depth):
            rgb = self.blocks_rgb[i](rgb)
            depth = self.blocks_depth[i](depth)
            rgb, depth = self.cmfrms[i]((rgb, depth))

        x = torch.cat([rgb, depth], dim=1)
        #x = torch.mean(x,(-1,-2))
        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x