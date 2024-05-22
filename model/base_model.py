import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
class ModuleHelper:
    @staticmethod
    def BNReLU(num_features, inplace=True):
        return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=inplace)
        )

    @staticmethod
    def BatchNorm2d(num_features):
        return nn.BatchNorm2d(num_features)

    @staticmethod
    def Conv3x3_BNReLU(in_channels, out_channels, stride=1, dilation=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                      groups=groups, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1_BNReLU(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ModuleHelper.BNReLU(out_channels)
        )

    @staticmethod
    def Conv1x1(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)
class Conv3x3(nn.Module):
    def __init__(self, in_chs, out_chs, dilation=1, dropout=None):
        super(Conv3x3, self).__init__()

        if dropout is None:
            self.conv = ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation)
        else:
            self.conv = nn.Sequential(
                ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation),
                nn.Dropout(dropout)
            )

        initialize_weights(self.conv)

    def forward(self, x):
        return self.conv(x)

class Dense_Aggregation_input_3(nn.Module):
    def __init__(self, chl):
        super(Dense_Aggregation_input_3, self).__init__()
        self.relu = nn.ReLU(True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4_1 = Conv3x3(chl, chl)
        self.conv4_2 = Conv3x3(chl, chl)
        self.conv4_3 = Conv3x3(chl, chl)

        self.conv3_1 = Conv3x3(chl, chl)
        self.conv3_2 = Conv3x3(chl*2, chl*2)

        self.conv_cat_4_3 = Conv3x3(chl * 2, chl * 2)
        self.conv_cat_3_2 = Conv3x3(chl * 3, chl * 3)

        self.conv_out = Conv3x3(chl * 3, chl * 3)

    def forward(self, x2, x3, x4):
        x4_1 = x4
        x4_2 = self.conv4_1(x4) * x3
        x4_3 = self.conv4_2(self.up2(x4)) * self.conv3_1(self.up2(x3)) * x2

        x43_cat = self.conv_cat_4_3(torch.cat((x4_2, self.conv4_3(x4_1)), 1))
        x32_cat = self.conv_cat_3_2(torch.cat((x4_3, self.conv3_2(self.up2(x43_cat))), 1))

        out = self.conv_out(x32_cat)

        return out

class Block_Resnet_GCN(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super(Block_Resnet_GCN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), )
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv21 = nn.Conv2d(in_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(out_channels, out_channels, bias=False, stride=stride,
                                kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0))
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.relu22 = nn.ReLU(inplace=True)


    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12(x1)
        x1 = self.relu12(x1)

        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu22(x2)

        x = x1 + x2
        return x


class BottleneckGCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_channels_gcn, stride=1):
        super(BottleneckGCN, self).__init__()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

        self.gcn = Block_Resnet_GCN(kernel_size, in_channels, out_channels_gcn)
        self.sigm = nn.Sigmoid()
        # self.conv1x1 = nn.Conv2d(out_channels_gcn, out_channels, 1, stride=stride, bias=False)
        # self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # identity = x
        # if self.downsample is not None:
        #     identity = self.downsample(identity)

        y = self.gcn(x)
        y = self.sigm(x)


        x = x * y
        return x

class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
        support = torch.matmul(x_t, self.weight)  # b x k x c

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ConcatNet(nn.Module):
    def __init__(self):
        super(ConcatNet, self).__init__()
        self.h = 176
        self.w = 176
        c1, c2 = 64, 128
        self.conv1 = nn.Sequential(nn.Conv2d(c1,c2,kernel_size=3,padding=1),nn.BatchNorm2d(c2),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(c2,c2,kernel_size=3,padding=1),nn.BatchNorm2d(c2),nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(2*c2,c2,kernel_size=3,padding=1),nn.BatchNorm2d(c2),nn.ReLU(),
                                   nn.Conv2d(c2,c2,kernel_size=1),nn.BatchNorm2d(c2),nn.ReLU())
    def forward(self,x1,x2):
        x1 = F.interpolate(x1, size=(self.h, self.w), mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.cat((x1,x2),dim=1)
        x = self.conv3(x)
        return x

class GraphNet(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.normalize_input = normalize_input

        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def init(self, initcache):
        if not os.path.exists(initcache):
            print(initcache + ' not exist!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                self.init_params(clsts, traindescs)
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        N = H*W
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype, layout=x.layout)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        soft_assign = F.softmax(soft_assign, dim=1)

        return soft_assign

    def forward(self, x):
        B, C, H, W = x.size()
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1) #across descriptor dim

        sigma = torch.sigmoid(self.sigma)
        soft_assign = self.gen_soft_assign(x, sigma) # B x C x N(N=HxW)
        #
        eps = 1e-9
        nodes = torch.zeros([B, self.node_num, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for node_id in range(self.node_num):
            residual = (x.view(B, C, -1).permute(0, 2, 1).contiguous() - self.anchor[node_id, :]).div(sigma[node_id, :]) # + eps)
            mm = soft_assign[:,node_id,:].unsqueeze(2)
            nodes[:, node_id, :] = residual.mul(soft_assign[:, node_id, :].unsqueeze(2)).sum(dim=1) / (soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps)

        nodes = F.normalize(nodes, p=2, dim=2) # intra-normalization
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1) # l2 normalize

        return nodes.view(B, C, self.node_num).contiguous(), soft_assign

class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim)
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert(loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous() # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x)) # b x c x k
        x = self.relu(x)
        return x

class MutualModule0(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(MutualModule0, self).__init__()
        self.gcn = CascadeGCNet(dim, loop=2)
        self.conv = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1,padding=0,bias=False),
                                  BatchNorm(dim),nn.ReLU(inplace=True))

    #graph0: edge, graph1/2: region, assign:edge
    def forward(self, edge_graph, region_graph1, region_graph2, assign):
        m = self.corr_matrix(edge_graph, region_graph1, region_graph2)
        edge_graph = edge_graph + m

        edge_graph = self.gcn(edge_graph)
        edge_x = edge_graph.bmm(assign) # reprojection
        edge_x = self.conv(edge_x.unsqueeze(3)).squeeze(3)
        return edge_x

    def corr_matrix(self, edge, region1, region2):
        assign = edge.permute(0, 2, 1).contiguous().bmm(region1)
        assign = F.softmax(assign, dim=-1) #normalize region-node
        m = assign.bmm(region2.permute(0, 2, 1).contiguous())
        m = m.permute(0, 2, 1).contiguous()
        return m


class ECGraphNet(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(ECGraphNet, self).__init__()
        self.dim = dim
        self.conv0 = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1,padding=0,bias=False),
                                  BatchNorm(dim),nn.ReLU(inplace=True))
        self.node_num = 32
        self.proj0 = GraphNet(self.node_num, self.dim, False)
        self.conv1 = nn.Sequential(nn.Conv2d(2*dim,dim,kernel_size=1,padding=0,bias=False),
                                  BatchNorm(dim),nn.ReLU(inplace=True))

    def forward(self, x, edge):
        b, c, h, w = x.shape
        device = x.device

        '''
        _, _, h1, w1 = edge.shape
        if h1 != h or w1 != w:
            edge = F.interpolate(edge, size=(h, w), mode='bilinear', align_corners=True)
        '''

        x1 = torch.sigmoid(edge).mul(x)  # elementwise-mutiply
        x1 = self.conv0(x1)
        nodes, _ = self.proj0(x1)  # b x c x k or b x k x c

        residual_x = x.view(b, c, -1).permute(2, 0, 1)[:, None] - nodes.permute(2, 0, 1)
        residual_x = residual_x.permute(2, 3, 1, 0).view(b, c, self.node_num, h, w).contiguous()

        '''
        residual_x = torch.zeros([b, c, self.node_num, h, w], device=device, dtype=x.dtype, layout=x.layout)
        for i in range(self.node_num):
            residual_x[:, :, i:i+1, :, :] = (x - nodes[:, :, i:i+1].unsqueeze(3)).unsqueeze(2)
        '''

        dists = torch.norm(residual_x, dim=1, p=2)  # b x k x h x w

        k = 5
        _, idx = torch.topk(dists, k=k, dim=1, largest=False)  # b x 5 x h x w

        num_points = h * w
        idx_base = torch.arange(0, b, device=device).view(-1, 1, 1, 1) * self.node_num
        idx = (idx + idx_base).view(-1)

        nodes = nodes.transpose(2, 1).contiguous()
        x1 = nodes.view(b * self.node_num, -1)[idx, :]
        x1 = x1.view(b, num_points, k, c)

        x2 = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(2).repeat(1, 1, k, 1)

        x1 = torch.cat((x1 - x2, x2), dim=3).permute(0, 3, 1, 2).contiguous()  # b x n x 5 x (2c)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x = x + x1.view(b, c, h, w)

        return x

class MutualModule1(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm2d, dropout=0.1):
        super(MutualModule1, self).__init__()
        self.dim = dim

        self.gcn = CascadeGCNet(dim, loop=3)

        self.pred0 = nn.Conv2d(self.dim, 1, kernel_size=1)  # predicted edge is used for edge-region mutual sub-module

        self.pred1_ = nn.Conv2d(self.dim, 1, kernel_size=1)  # region prediction

        # conv region feature afger reproj
        self.conv0 = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1,padding=0,bias=False),
                                  BatchNorm(dim),nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=1,padding=0,bias=False),
                                  BatchNorm(dim),nn.ReLU(inplace=True))

        self.ecg = ECGraphNet(self.dim, BatchNorm, dropout)

    def forward(self, region_x, region_graph, region_graph3, assign, edge_graph):
        # b, c, h, w = edge_graph.shape

        region_assign = region_graph.permute(0, 2, 1).contiguous().bmm(edge_graph)
        region_assign = F.softmax(region_assign,dim=-1)
        m = region_assign.bmm(region_graph3.permute(0,2,1).contiguous())
        m = m.permute(0,2,1).contiguous()
        region_graph = region_graph + m


        region_graph = self.gcn(region_graph)
        n_region_x = region_graph.bmm(assign)
        n_region_x = self.conv0(n_region_x.view(region_x.size()))

        region_x = region_x + n_region_x  # raw-feature with residual

        region_x = self.conv1(region_x)

        # enhance
        # region_x = self.ecg(region_x, edge)

        region = self.pred1_(region_x)

        return region_x, region