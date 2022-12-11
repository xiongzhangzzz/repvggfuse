import torch.nn as nn
import numpy as np
import torch
import copy
import torch.nn.functional as F
from cnsn import CrossNorm
EPSILON = 1e-10


# addition fusion strategy
def addition_fusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2):
    # avg, max, nuclear
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f = f_spatial
    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
    result.add_module('bn', CrossNorm(out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = CrossNorm(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, CrossNorm)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

 
def make_stage(inplanes, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    blocks = []
    if_first_layer = 1
    for stride in strides:
        if if_first_layer == 1:
            blocks.append(RepVGGBlock(inplanes, planes, kernel_size=3,
                                      stride=stride, padding=1))
            if_first_layer -= 1                 
        else:
            blocks.append(RepVGGBlock(planes, planes, kernel_size=3,
                                      stride=stride, padding=1))
    return nn.Sequential(*blocks)   


class RepVGG_Fuse_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_blocks=[2,4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], deploy=False):
        super(RepVGG_Fuse_net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.width_multiplier = width_multiplier
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.deploy = deploy

        # encoder
        self.stage0 = RepVGGBlock(self.in_channels, out_channels=self.in_planes, kernel_size=3, stride=1)
        self.stage1 = make_stage(self.in_planes, int(64 * self.width_multiplier[0]), self.num_blocks[0], stride=1)
        self.stage2 = make_stage(int(64 * self.width_multiplier[0]), int(128 * self.width_multiplier[1]), self.num_blocks[1], stride=1)
        self.stage3 = make_stage(int(128 * self.width_multiplier[1]), int(256 * self.width_multiplier[2]), self.num_blocks[2], stride=1)
        self.stage4 = make_stage(int(256 * self.width_multiplier[2]), int(512 * self.width_multiplier[3]), self.num_blocks[3], stride=1)
        # decoder
        self.conv0 = nn.Conv2d(int(512 * self.width_multiplier[3]), int(256 * self.width_multiplier[2]),
                        kernel_size=3, stride=1,padding=1)
        self.conv1 = nn.Conv2d(int(256 * self.width_multiplier[2]), int(128 * self.width_multiplier[1]),
                        kernel_size=3, stride=1,padding=1)    
        self.conv2 = nn.Conv2d(int(128 * self.width_multiplier[1]), int(64 * self.width_multiplier[0]),
                        kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(int(64 * self.width_multiplier[0]), self.out_channels,
                        kernel_size=3, stride=1,padding=1)


    def encoder(self, input):
        x1 = self.stage0(input)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        x5 = self.stage4(x4)
        return [x5]

    def fusion(self, en1, en2, strategy_type='addition'):
        # addition
        if strategy_type == 'attention_weight':
            # attention weight
            fusion_function = attention_fusion_weight
        elif strategy_type == 'addition':
            fusion_function = addition_fusion
    
        f_0 = fusion_function(en1[0], en2[0])
        return [f_0]

    def decoder(self, f_en):
        x1 = self.conv0(f_en[0])
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        output = self.conv3(x3)

        return [output]

