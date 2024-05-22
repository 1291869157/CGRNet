import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from model.vgg import B2_VGG
from model.base_model import *


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Sal_Module(nn.Module):

    def __init__(self, in_fea=[256, 512, 512], mid_fea=64):
        super(Sal_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        # self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self, x3, x4, x5):
        _, _, h, w = x3.size()
        sal3_fea = self.relu(self.conv3(x3))
        sal3 = self.relu(self.conv5_2(sal3_fea))
        sal4_fea = self.relu(self.conv4(x4))
        sal4 = self.relu(self.conv5_4(sal4_fea))
        sal5_fea = self.relu(self.conv5(x5))
        sal5 = self.relu(self.conv5_5(sal5_fea))

        sal4 = F.interpolate(sal4, size=(h, w), mode='bilinear', align_corners=True)
        sal5 = F.interpolate(sal5, size=(h, w), mode='bilinear', align_corners=True)

        sal = torch.cat([sal3, sal4, sal5], dim=1)
        sal = self.rcab(sal)

        return sal

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.asppconv = torch.nn.Sequential()
        if bn_start:
            self.asppconv = nn.Sequential(
                nn.BatchNorm2d(input_num),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        else:
            self.asppconv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],
                                      mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class Back_VGG(nn.Module):
    def __init__(self, channel=32):
        super(Back_VGG, self).__init__()
        self.vgg = B2_VGG()
        self.sal_proj0 = GraphNet(node_num = 4, dim=128, normalize_input=False)
        self.edge_proj0 = GraphNet(node_num = 4, dim=128, normalize_input=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 512)
        self.relu = nn.ReLU(True)
        self.sal_layer = Sal_Module()
        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 32,
                                                       output_stride=16)
        self.classification = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=3,padding=1)
        )

        self.ConcatNet = ConcatNet()
        self.init_edgemap = nn.Conv2d(1,512,kernel_size=1)
        self.nonloacl = BottleneckGCN(1,1,5,1)
        self.edge_gconv = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.salg_conv1 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.salg_conv2 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.salg_conv3 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=False), nn.ReLU(inplace=True))
        self.repro1 = MutualModule0(128)
        self.repro2 = MutualModule1(128)
        self.init_adj = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.adj_salgmap = nn.Sequential(nn.Conv2d(192,128,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                                    nn.Conv2d(128,128,kernel_size=3,padding=1),nn.ReLU(inplace=True))
        self.region_conv = nn.Conv2d(1,channel,kernel_size=3,padding=1,bias=False)

        self.dense_agg3 = Dense_Aggregation_input_3(channel)

        self.rcab_feat = RCAB(channel * 6)
        self.sal_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.edge_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(channel*3)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel*2)
        self.after_aspp_conv5 = nn.Conv2d(channel*6, channel, kernel_size=1, bias=False)
        self.after_aspp_conv2 = nn.Conv2d(128, channel, kernel_size=1, bias=False)
        self.final_sal_seg = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1, bias=False))
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.fused_edge_sal = nn.Conv2d(96, 1, kernel_size=3, padding=1, bias=False)


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x_size = x.size()
        x1 = self.vgg.conv1(x) ## 352*352*64
        x2 = self.vgg.conv2(x1)  ## 176*176*128
        x3 = self.vgg.conv3(x2)   ## 88*88*256
        x4 = self.vgg.conv4(x3)  ## 44*44*512
        x5 = self.vgg.conv5(x4)  ## 22*22*512
        # x_size = x3.size()

        edge_gmap = self.ConcatNet(x1,x2)
        sal_map = self.sal_layer(x3,x4,x5)
        sal_gmap = F.interpolate(sal_map, (176,176), mode='bilinear')
        sal_gmap = self.adj_salgmap(sal_gmap)


        edge_graph, edge_assign = self.edge_proj0(edge_gmap)
        sal_graph, sal_assign = self.sal_proj0(sal_gmap)

        edge_graph = self.edge_gconv(edge_graph)
        sal_graph1 = self.salg_conv1(sal_graph)
        sal_graph2 = self.salg_conv2(sal_graph)
        sal_graph3 = self.salg_conv3(sal_graph)
        n_edge_x = self.repro1(edge_graph,sal_graph1,sal_graph2,edge_assign)
        edge_gmap = edge_gmap + n_edge_x.view(edge_gmap.size()).contiguous()
        region_x, region = self.repro2(sal_gmap,sal_graph,sal_graph3,sal_assign,edge_graph)

        edge_gmap = self.init_adj(edge_gmap)
        edge_gmap = F.interpolate(edge_gmap,x_size[2:],mode='bilinear')


        ####
        # edge_map = self.edge_layer(x1, x2, x3)
        edge_gmap = self.nonloacl(edge_gmap)
        edge_out = torch.sigmoid(edge_gmap)
        im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        cat = torch.cat((edge_out, canny), dim=1)
        acts = self.fuse_canny_edge(cat)
        acts = torch.sigmoid(acts)

        x5 = self.aspp(x5, acts)
        x_conv5 = self.after_aspp_conv5(x5)
        x_conv2 = self.after_aspp_conv2(x2)
        x_conv5_up = F.interpolate(x_conv5, x2.size()[2:], mode='bilinear', align_corners=True)
        feat_fuse = torch.cat([x_conv5_up, x_conv2], 1)

        sal_init = self.final_sal_seg(feat_fuse)



        sal_feature = self.sal_conv(sal_init)
        edge_feature = self.edge_conv(edge_gmap)
        region_feature = self.region_conv(region)
        # edge_feature = torch.cat((edge_feature,region_feature),1)
        # sal_edge_feature = self.relu(torch.cat((sal_feature, edge_feature), 1))
        sal_edge_map_feature = self.dense_agg3(edge_feature,sal_feature,region_feature)
        sal_edge_feature = self.rcab_sal_edge(sal_edge_map_feature)
        sal_ref = self.fused_edge_sal(sal_edge_feature)
        sal_init = F.interpolate(sal_init, x_size[2:], mode='bilinear')
        return sal_init, edge_gmap, sal_ref