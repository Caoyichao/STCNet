from torch import nn
import torch.nn.functional as TNF

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
import torch.nn.init as init
from torch.nn.init import normal_, constant_
import senet
import senet_branch


BN_MOMENTUM = 0.1

def initialize_weights(net_l, scale=1.):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class STCNet(nn.Module):
    def __init__(self, num_class = 2, num_segments = 8, modality = "RGB",
                 base_model='se_resnext50_32x4d',branch_model='se_resnext50_32x4d', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5,img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True):
        super(STCNet, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec == True:
            print(("""
    Initializing STCNet with base model: {}.
    STCNet Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)
        self._prepare_branch_model(branch_model) # initialize flow/residual branch

        self.conv_fuse_RGB = nn.Conv2d(2048, 256, 1, 1, bias=True)
        self.conv_fuse_FLOW = nn.Conv2d(2048, 256, 1, 1, bias=True)

        self.cross_down = [nn.Sequential(
                        nn.Conv2d(channels, channels, 1, 1, bias=True),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=False)).cuda()
                         for channels in [256, 512, 1024, 2048]]
        self.cross_up = [nn.Sequential(
                        nn.Conv2d(channels, channels, 1, 1, bias=True),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=False)).cuda()
                         for channels in [256, 512, 1024, 2048]]
        self.up_path = [nn.Sequential(
                        nn.Conv2d(channels, channels, 1, 1, bias=True),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=False)).cuda()
                         for channels in [256, 512, 1024, 2048]]
        self.down_path = [nn.Sequential(
                        nn.Conv2d(channels, channels, 1, 1, bias=True),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=False)).cuda()
                         for channels in [256, 512, 1024, 2048]]

        self.cls_module = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.last_fc = nn.Linear(256, num_class)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def forward(self, RGB_input, FLOW_input):
        sample_len = (3)

        #[64,56,56]
        feature_x0 = self.base_model.layer0(RGB_input.view((-1, sample_len) + RGB_input.size()[-2:]))
        branch_x0 = self.branch_model.layer0(FLOW_input.view((-1, sample_len) + FLOW_input.size()[-2:]))
        #[256,56,56]
        feature_x1 = self.base_model.layer1(feature_x0)
        branch_x1 = self.branch_model.layer1(branch_x0)
        #branch_x1_up = self.cross_up[0](branch_x1)
        #branch_x1_ = self.down_path[0](branch_x1)
        #feature_x1_down = self.cross_down[0](feature_x1)
        #feature_x1_ = self.up_path[0](feature_x1)
        feature_x1_merge = feature_x1 + branch_x1
        branch_x1_merge = branch_x1 + feature_x1
        #[512,28,28]
        feature_x2 = self.base_model.layer2(feature_x1_merge)
        branch_x2 = self.branch_model.layer2(branch_x1_merge)
        #branch_x2_up = self.cross_up[1](branch_x2)
        #branch_x2_ = self.down_path[1](branch_x2)
        #feature_x2_down = self.cross_down[1](feature_x2)
        #feature_x2_ = self.up_path[1](feature_x2)
        feature_x2_merge = feature_x2 + branch_x2
        branch_x2_merge = feature_x2 + branch_x2
        #[1024,14,14]
        feature_x3 = self.base_model.layer3(feature_x2_merge)
        branch_x3 = self.branch_model.layer3(branch_x2_merge)
        #branch_x3_up = self.cross_up[2](branch_x3)
        #branch_x3_ = self.down_path[2](branch_x3)
        #feature_x3_down = self.cross_down[2](feature_x3)
        #feature_x3_ = self.up_path[2](feature_x3)
        feature_x3_merge = feature_x3 + branch_x3
        branch_x3_merge = feature_x3 + branch_x3
        #[2048,7,7]
        feature_x4 = self.base_model.layer4(feature_x3_merge)
        branch_x4 = self.branch_model.layer4(branch_x3_merge)
        # branch_x4_up = self.cross_up[3](branch_x4)
        # feature_x4_down = self.cross_down[3](feature_x4)
        # feature_x4_ = feature_x4 + branch_x4_up
        # branch_x4_ = branch_x4 + feature_x4_down
        #[256,7,7]
        RGB_out = TNF.relu(self.conv_fuse_RGB(feature_x4 + branch_x4), inplace=False)
        B, C, H, W = RGB_out.size()
        RGB_out = RGB_out.view(-1, self.num_segments * C, H, W)

        cls_out = self.cls_module(RGB_out)
        cls_out = cls_out.view(-1, 256)
        cls_out = self.last_fc(cls_out)

        return cls_out#, ret_feature



    def _prepare_STCNet(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.consensus_type in ['TRN','TRNmultiscale']:
                # create a new linear layer as the frame feature
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                # the default consensus types in STCNet
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim


    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'resnext' in base_model:
            self.base_model = getattr(senet, base_model)(num_classes=1000, pretrained='imagenet')
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
    
    def _prepare_branch_model(self, branch_model):

        if 'resnet' in branch_model or 'resnext' in branch_model:
            self.branch_model = getattr(senet, branch_model)(num_classes=1000, pretrained='imagenet')
            self.branch_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('Unknown base model: {}'.format(branch_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(STCNet, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        transpose = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.ConvTranspose2d):
                transpose.extend(m.parameters())
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': transpose, 'lr_mult': 1, 'decay_mult': 0,
             'name': 'conv transpose'}
        ]

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        #return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
        #                                           GroupRandomHorizontalFlip()])
        rrc = RandomResizedCrop(size=224, scale=(0.9, 1), ratio=(3./4., 4./3.))
        rp = RandomPerspective(anglex=3, angley=3, anglez=3, shear=3)
        # Improve generalization
        rhf = RandomHorizontalFlip(p=0.5)
        # Deal with dirts, ants, or spiders on the camera lense
        re = RandomErasing(p=0.5, scale=(0.003, 0.01), ratio=(0.3, 3.3), value=0)
        cj = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(-0.1, 0.1), gamma=0.3)

        return torchvision.transforms.Compose([cj, rrc, rp, rhf])
        #return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .9,]),
        #                                           GroupRandomHorizontalFlip()])


if __name__ == '__main__':
    import torch
    RGB_input = torch.zeros([1,8,3,224,224])
    FLOW_input = torch.zeros([1,8,3,224,224])
    model = STCNet()
    model = model.cuda()
    print(model)
    RGB_input = RGB_input.cuda()
    FLOW_input = FLOW_input.cuda()
    res = model(RGB_input,FLOW_input)
    print(res.shape)
