#############
import torch
import torch.nn as nn
import torch.nn.functional as F

#############

# Contains:
# - The NN models


class PointNetfeat(nn.Module):
    def __init__(self, n_in=3, n_layers=4, size_layers=128, global_feat=True, feature_transform=False, normalize=False):
        super(PointNetfeat, self).__init__()
        self.n_layers = n_layers
        self.size_layers = size_layers
        self.normalize = normalize
        self.conv1 = torch.nn.Conv1d(n_in, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv41 = torch.nn.Conv1d(128, size_layers, 1)
        # self.conv42 = torch.nn.Conv1d(128, 128, 1)
        self.convs = []
        self.bn = []
        for i in range(0, n_layers):
            self.convs.append(torch.nn.Conv1d(size_layers, size_layers, 1))
            self.bn.append(nn.BatchNorm1d(size_layers))

        self.conv5 = torch.nn.Conv1d(size_layers, 1024, 1)
        self.dense1 = torch.nn.Linear(1024, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn41 = nn.BatchNorm1d(size_layers)
        self.bn42 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        for i in range(0, self.n_layers):
            x = F.relu(self.bn[i](self.convs[i](x)))
        x = F.relu(self.bn5(self.conv5(x)))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn6(self.dense1(x)))
        x = F.relu(self.bn7(self.dense2(x)))

        trans_feat = None
        trans = None
        x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetBasis(nn.Module):
    def __init__(self, n_basis=20, n_layers=4, size_layers=128, n_in=3, normalize=False, feature_transform=False):
        super(PointNetBasis, self).__init__()
        self.k = n_basis
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            n_in=n_in,
            n_layers=n_layers,
            size_layers=size_layers,
            normalize=False,
            global_feat=False,
            feature_transform=feature_transform,
        )
        self.conv1 = torch.nn.Conv1d(1280, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv2b = torch.nn.Conv1d(256, 256, 1)
        self.conv2c = torch.nn.Conv1d(256, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn2b = nn.BatchNorm1d(256)
        self.bn2c = nn.BatchNorm1d(256)
        self.m = nn.Dropout(p=0.3)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = F.relu(self.bn2c(self.conv2c(x)))
        x = self.m(x)
        # x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv3(x)
        x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


#############


class TemplateNet(nn.Module):
    def __init__(self, n_points=1000, n_basis=20, batch_norm=False):
        super(TemplateNet, self).__init__()
        self.batch_norm = batch_norm
        self.n_basis = n_basis

        self.n_points = n_points
        self.dense_list = [1024, 512, 256, 512, 1024]
        self.dense_layers = []
        self.bn_layers = []

        self.dense_layers.append(torch.nn.Linear(n_points * 3, self.dense_list[0]).cuda())

        if batch_norm:
            self.bn_layers.append(nn.BatchNorm1d(self.dense_list[0]).cuda())

        for i in range(len(self.dense_list[1:])):
            self.dense_layers.append(torch.nn.Linear(self.dense_list[i - 1], self.dense_list[i]).cuda())
            if batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(self.dense_list[i]).cuda())

        self.dense_layers.append(torch.nn.Linear(self.dense_list[i], n_points * 20).cuda())

    def forward(self, x):
        x = x.reshape(-1, self.n_points * x.shape[1])

        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            if self.batch_norm:
                x = self.bn_layers[i](x)

        x = self.dense_layers[-1](x)

        x = x.reshape(-1, self.n_points, self.n_basis)
        return x, None, None


#############
import numpy as np


class PointNetGlob(nn.Module):
    def __init__(
        self,
        n_points=500,
        n_in=3,
        n_layers=4,
        size_layers=128,
        global_feat=True,
        feature_transform=False,
        normalize=False,
        b_min=np.array([-1.2, -1.4, -1.2]),
        b_max=np.array([1.2, 1.3, 1.2]),
    ):
        super(PointNetGlob, self).__init__()
        self.b_min = torch.FloatTensor(b_min).cuda()
        self.b_max = torch.FloatTensor(b_max).cuda()
        self.bb = self.b_max - self.b_min

        self.n_layers = n_layers
        self.size_layers = size_layers
        self.n_points = n_points
        self.normalize = normalize
        self.conv1 = torch.nn.Conv1d(n_in, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv41 = torch.nn.Conv1d(128, size_layers, 1)

        self.convs = []
        self.bn = []
        for i in range(0, n_layers):
            self.convs.append(torch.nn.Conv1d(size_layers, size_layers, 1).cuda())
            self.bn.append(nn.BatchNorm1d(size_layers).cuda())

        self.conv5 = torch.nn.Conv1d(size_layers, 1024, 1)
        self.dense1 = torch.nn.Linear(1024, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn41 = nn.BatchNorm1d(size_layers)
        self.bn42 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            n_in=n_in,
            n_layers=n_layers,
            size_layers=size_layers,
            normalize=False,
            global_feat=False,
            feature_transform=feature_transform,
        )

        hidden_dim = 1024
        output_dim = self.n_points * 3
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(256 + 3, hidden_dim, 1)).cuda()
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim * 2, 1)).cuda()

        self.query_layers = []
        for i in range(0, n_layers):
            self.query_layers.append(nn.utils.weight_norm(nn.Conv1d(hidden_dim * 2, hidden_dim * 2, 1)).cuda())
        
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim * 2, output_dim, 1)).cuda()
        self.actvn = nn.ReLU()

    def forward(self, x):
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        for i in range(0, self.n_layers):
            x = F.relu(self.bn[i](self.convs[i](x)))
        x = F.relu(self.bn5(self.conv5(x)))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn6(self.dense1(x)))
        x = F.relu(self.bn7(self.dense2(x)))

        trans_feat = None
        trans = None
        x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
        self.feats = x

    def query(self, x):
        _B, _numpoints, _ = x.shape

        normalized_p = x
        point_features = normalized_p.permute(0, 2, 1)
        point_features = torch.cat([point_features, self.feats], 1)

        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        for i in range(0, self.n_layers):
            point_features = self.actvn(self.query_layers[i](point_features))
 
        point_features = self.fc_out(point_features)

        return point_features


########################
import functools

class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                m.weight.data.normal_(0.0, 0.02)
            except:
                for i in m.children():
                    i.apply(self._weights_init_fn)
                return
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer


class Network_LVD(NetworkBase):
    def __init__(self, hidden_dim=512, output_dim=6890, res=128, input_dim=1, 
                 b_min = np.array([-1.2, -1.4, -1.2]), b_max = np.array([1.2, 1.3, 1.2]), 
                 paradigm='LVD',selfsup=False, device='cuda', power_feat=False):
        super(Network_LVD, self).__init__()
        self._name = 'voxel_encoder'
        self.res = res 
        self.paradigm = paradigm
        self.b_min = torch.FloatTensor(b_min).to(device)
        self.b_max = torch.FloatTensor(b_max).to(device)
        self.bb = self.b_max - self.b_min
        self.selfsup = selfsup
        self.output_dim = output_dim
        # if power_feat:
        #     self.conv_1 = nn.utils.weight_norm(nn.Conv3d(input_dim, 32, 3, stride=2, padding=1))  # out: 32
            
            
            
        # else:
        self.conv_1 = nn.utils.weight_norm(nn.Conv3d(input_dim, 32, 3, stride=2, padding=1))  # out: 32
        self.conv_1_1 = nn.utils.weight_norm(nn.Conv3d(32, 32, 3, padding=1))  # out: 32
        self.conv_2 = nn.utils.weight_norm(nn.Conv3d(32, 64, 3, padding=1))  # out: 16
        self.conv_2_1 = nn.utils.weight_norm(nn.Conv3d(64, 64, 3, padding=1))  # out: 16
        self.conv_3 = nn.utils.weight_norm(nn.Conv3d(64, 96, 3, padding=1))  # out: 8
        self.conv_3_1 = nn.utils.weight_norm(nn.Conv3d(96, 96, 3, padding=1))  # out: 8
        self.conv_4 = nn.utils.weight_norm(nn.Conv3d(96, 128, 3, padding=1))  # out: 8
        self.conv_4_1 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_5 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_5_1 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_6 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_6_1 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_7 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_7_1 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8

        if res == 128:
            feature_size = (3 + input_dim + 32 + 64 + 96 + 128 + 128 + 128 + 128)
        else:
            feature_size = (3 + input_dim + 32 + 64 + 96 + 128 + 128 + 128)
            
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_3 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_4 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, output_dim, 1))
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)
        
        if self.selfsup == True:
            self.fc_ss_0 = nn.utils.weight_norm(nn.Linear(128, hidden_dim))
            self.fc_ss_1 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim*2))
            self.fc_ss_2 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))
            self.fc_ss_3 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))
            self.fc_ss_4 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))            
            self.fc_ss_out = nn.utils.weight_norm(nn.Linear(hidden_dim*2, self.output_dim))  
            #self.fc_ss_out = nn.utils.weight_norm(nn.Linear(hidden_dim*2, 256))
            
    def forward(self, x):
        features0 = x
        x = self.actvn(self.conv_1(x))
        x = self.actvn(self.conv_1_1(x))
        features1 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_2(x))
        x = self.actvn(self.conv_2_1(x))
        features2 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_3(x))
        x = self.actvn(self.conv_3_1(x))
        features3 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_4(x))
        x = self.actvn(self.conv_4_1(x))
        features4 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_5(x))
        x = self.actvn(self.conv_5_1(x))
        features5 = x
        x = self.maxpool(x)

        if self.res == 128:
            x = self.actvn(self.conv_6(x))
            x = self.actvn(self.conv_6_1(x))
            features6 = x
            x = self.maxpool(x)

            x = self.actvn(self.conv_7(x))
            x = self.actvn(self.conv_7_1(x))
            features7 = x

            self.features = [features0, features1, features2, features3, features4, features5, features6, features7]
        
        if self.res == 64:
            x = self.actvn(self.conv_7(x))
            x = self.actvn(self.conv_7_1(x))
            features7 = x

            self.features = [features0, features1, features2, features3, features4, features5, features7]            
        
    def query(self, p):
        _B, _numpoints, _ = p.shape

        normalized_p = (p - self.b_min)/self.bb*2 - 1
        point_features = normalized_p.permute(0, 2, 1)
        for j, feat in enumerate(self.features):
            interpolation = F.grid_sample(feat, normalized_p.unsqueeze(1).unsqueeze(1), align_corners=False).squeeze(2).squeeze(2)
            point_features = torch.cat((point_features, interpolation), 1)

        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        point_features = self.actvn(self.fc_3(point_features))
        point_features = self.actvn(self.fc_4(point_features))
        point_features = self.fc_out(point_features)

        return point_features

    def self_sup(self):
        f1 = self.actvn(self.fc_ss_0(torch.squeeze(self.features[6])))
        f1 = self.actvn(self.fc_ss_1(f1))
        f1 = self.fc_ss_2(f1)
        f1 = self.fc_ss_3(f1)
        f1 = self.fc_ss_4(f1)
        f1 = self.fc_ss_out(f1)
        
        return f1
#################################

class Network_LVD_PowerUP(NetworkBase):
    def __init__(self, hidden_dim=512, output_dim=6890, res=128, input_dim=1, 
                 b_min = np.array([-1.2, -1.4, -1.2]), b_max = np.array([1.2, 1.3, 1.2]), 
                 paradigm='LVD',selfsup=False, segm=0, labels=[], unsup=False, device='cuda'):
        super(Network_LVD_PowerUP, self).__init__()
        self._name = 'voxel_encoder'
        self.res = res 
        self.paradigm = paradigm
        self.b_min = torch.FloatTensor(b_min).to(device)
        self.b_max = torch.FloatTensor(b_max).to(device)
        self.bb = self.b_max - self.b_min
        self.selfsup = selfsup
        self.output_dim = output_dim
        self.segm = segm
        self.labels = labels 
        self.unsup = unsup 
        
                # torch.Size([4, 13, 64, 64, 64])      
        # else:
        self.conv_1 = nn.utils.weight_norm(nn.Conv3d(input_dim, 32*2, 3, stride=2, padding=1))  # out: 32
        self.conv_1_1 = nn.utils.weight_norm(nn.Conv3d(32*2, 32*2, 3, padding=1))  # out: 32
        self.conv_2 = nn.utils.weight_norm(nn.Conv3d(32*2, 64*2, 3, padding=1))  # out: 16
        self.conv_2_1 = nn.utils.weight_norm(nn.Conv3d(64*2, 64*2, 3, padding=1))  # out: 16
        self.conv_3 = nn.utils.weight_norm(nn.Conv3d(64*2, 96*2, 3, padding=1))  # out: 8
        self.conv_3_1 = nn.utils.weight_norm(nn.Conv3d(96*2, 96*2, 3, padding=1))  # out: 8
        self.conv_4 = nn.utils.weight_norm(nn.Conv3d(96*2, 128*2, 3, padding=1))  # out: 8
        self.conv_4_1 = nn.utils.weight_norm(nn.Conv3d(128*2, 128*2, 3, padding=1))  # out: 8
        self.conv_5 = nn.utils.weight_norm(nn.Conv3d(128*2, 128*2, 3, padding=1))  # out: 8
        self.conv_5_1 = nn.utils.weight_norm(nn.Conv3d(128*2, 128*2, 3, padding=1))  # out: 8
        self.conv_6 = nn.utils.weight_norm(nn.Conv3d(128*2, 128*2, 3, padding=1))  # out: 8
        self.conv_6_1 = nn.utils.weight_norm(nn.Conv3d(128*2, 128*2, 3, padding=1))  # out: 8
        self.conv_7 = nn.utils.weight_norm(nn.Conv3d(128*2, 128*2, 3, padding=1))  # out: 8
        self.conv_7_1 = nn.utils.weight_norm(nn.Conv3d(128*2, 128*2, 3, padding=1))  # out: 8

        
        
        if res == 128:
            feature_size = (3 + input_dim + 32 + 64 + 96 + 128 + 128 + 128 + 128)
        else:
            feature_size = (3 + input_dim + 32*2 + 64*2 + 96*2 + 128*2 + 128*2 + 128*2)
        
        
        
        
        if segm==0:    
            self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
            self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
            self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
            self.fc_3 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
            self.fc_4 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
            self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, output_dim, 1))
            self.actvn = nn.ReLU()
        else:
            self.segm_list = nn.ModuleList()
            self.selections = []
            self.part_idxs = []
            
            self.actvn = nn.ReLU()
            for i in range(segm):
                self.selections.append(labels==i)
                self.part_idxs.append(np.where(self.labels==i))
                self.segm_list.append(nn.ModuleList([nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1)).to(device),
                                nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1)).to(device),
                                nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1)).to(device),
                                nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1)).to(device),
                                nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1)).to(device),
                                nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, np.sum(self.selections[i])*3, 1).to(device))
                                ]))
            self.parts_order = np.argsort(np.squeeze(np.hstack(self.part_idxs)))
            
        self.maxpool = nn.MaxPool3d(2)
        
        if self.selfsup == True:
            self.fc_ss_0 = nn.utils.weight_norm(nn.Linear(128, hidden_dim))
            self.fc_ss_1 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim*2))
            self.fc_ss_2 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))
            self.fc_ss_3 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))
            self.fc_ss_4 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))            
            self.fc_ss_out = nn.utils.weight_norm(nn.Linear(hidden_dim*2, self.output_dim))  
            #self.fc_ss_out = nn.utils.weight_norm(nn.Linear(hidden_dim*2, 256))
            
        self.layers_first = [self.conv_1,
                        self.conv_2, self.conv_3, self.conv_4, 
                        self.conv_5, self.conv_7]
        
        self.layers_second = [self.conv_1_1,
                        self.conv_2_1, self.conv_3_1, self.conv_4_1, 
                        self.conv_5_1, self.conv_7_1]
                    
    def forward(self, x):
        features = [] 
        features.append(x)
        for i in range(len(self.layers_first)):
            x = self.actvn(self.layers_first[i](x))
            x = self.actvn(self.layers_second[i](x))
            features.append(x)
            if i<len(self.layers_first)-1:
                x = self.maxpool(x)
            
        # features0 = x
        
        # x = self.actvn(self.conv_1(x))
        # x = self.actvn(self.conv_1_1(x))
        # features1 = x
        # x = self.maxpool(x)

        # x = self.actvn(self.conv_2(x))
        # x = self.actvn(self.conv_2_1(x))
        # features2 = x
        # x = self.maxpool(x)

        # x = self.actvn(self.conv_3(x))
        # x = self.actvn(self.conv_3_1(x))
        # features3 = x
        # x = self.maxpool(x)

        # x = self.actvn(self.conv_4(x))
        # x = self.actvn(self.conv_4_1(x))
        # features4 = x
        # x = self.maxpool(x)

        # x = self.actvn(self.conv_5(x))
        # x = self.actvn(self.conv_5_1(x))
        # features5 = x
        # x = self.maxpool(x)

        # if self.res == 128:
        #     x = self.actvn(self.conv_6(x))
        #     x = self.actvn(self.conv_6_1(x))
        #     features6 = x
        #     x = self.maxpool(x)

        #     x = self.actvn(self.conv_7(x))
        #     x = self.actvn(self.conv_7_1(x))
        #     features7 = x

        #     self.features = [features0, features1, features2, features3, features4, features5, features6, features7]
        
        # if self.res == 64:
        #     x = self.actvn(self.conv_7(x))
        #     x = self.actvn(self.conv_7_1(x))
        #     features7 = x

        self.features = [features[i] for i in range(len(features))]
                    
    def query(self, p, labels=[]):
        _B, _numpoints, _ = p.shape

        normalized_p = (p - self.b_min)/self.bb*2 - 1
        point_features = normalized_p.permute(0, 2, 1)
        for j, feat in enumerate(self.features):
            interpolation = F.grid_sample(feat, normalized_p.unsqueeze(1).unsqueeze(1), align_corners=False).squeeze(2).squeeze(2)
            point_features = torch.cat((point_features, interpolation), 1)

        if self.unsup:
            dist = point_features[:,12]
        
        if self.segm==0:
            point_features = self.actvn(self.fc_0(point_features))
            point_features = self.actvn(self.fc_1(point_features))
            point_features = self.actvn(self.fc_2(point_features))
            point_features = self.actvn(self.fc_3(point_features))
            point_features = self.actvn(self.fc_4(point_features))
            point_features = self.fc_out(point_features)
        else:
            list_point_features = []
            for i in range(self.segm):
                point_features_loc = point_features
                point_features_loc = self.actvn(self.segm_list[i][0](point_features_loc))
                point_features_loc = self.actvn(self.segm_list[i][1](point_features_loc))
                point_features_loc = self.actvn(self.segm_list[i][2](point_features_loc))
                point_features_loc = self.actvn(self.segm_list[i][3](point_features_loc))
                point_features_loc = self.actvn(self.segm_list[i][4](point_features_loc))
                point_features_loc = self.segm_list[i][5](point_features_loc)
                list_point_features.append(point_features_loc.reshape(p.shape[0],-1,3,p.shape[1]))
            point_features = torch.cat(list_point_features,axis=1)
            point_features = point_features[:, self.parts_order, :]#.reshape(p.shape[0],-1,p.shape[1])
        
        if self.unsup:
            return point_features, dist 
        else:
            return point_features

    def self_sup(self):
        f1 = self.actvn(self.fc_ss_0(torch.squeeze(self.features[6])))
        f1 = self.actvn(self.fc_ss_1(f1))
        f1 = self.fc_ss_2(f1)
        f1 = self.fc_ss_3(f1)
        f1 = self.fc_ss_4(f1)
        f1 = self.fc_ss_out(f1)
        
        return f1
##################


class Network_LVD_PowerUP2(NetworkBase):
    def __init__(self, hidden_dim=512, output_dim=6890, res=128, input_dim=1, 
                 b_min = np.array([-1.2, -1.4, -1.2]), b_max = np.array([1.2, 1.3, 1.2]), 
                 paradigm='LVD',selfsup=False, power_factor = 5, device='cuda'):
        super(Network_LVD_PowerUP2, self).__init__()
        self._name = 'voxel_encoder'
        self.res = res 
        self.paradigm = paradigm
        self.b_min = torch.FloatTensor(b_min).to(device)
        self.b_max = torch.FloatTensor(b_max).to(device)
        self.bb = self.b_max - self.b_min
        self.selfsup = selfsup
        self.output_dim = output_dim


                # torch.Size([4, 13, 64, 64, 64])
        factor = power_factor        
        # else:
        self.conv_1 = nn.utils.weight_norm(nn.Conv3d(input_dim, 32*factor, 3, stride=2, padding=1))  # out: 32
        self.conv_1_1 = nn.utils.weight_norm(nn.Conv3d(32*factor, 32*factor, 3, padding=1))  # out: 32
        self.conv_2 = nn.utils.weight_norm(nn.Conv3d(32*factor, 64*factor, 3, padding=1))  # out: 16
        self.conv_2_1 = nn.utils.weight_norm(nn.Conv3d(64*factor, 64*factor, 3, padding=1))  # out: 16
        self.conv_3 = nn.utils.weight_norm(nn.Conv3d(64*factor, 96*factor, 3, padding=1))  # out: 8
        self.conv_3_1 = nn.utils.weight_norm(nn.Conv3d(96*factor, 96*factor, 3, padding=1))  # out: 8
        self.conv_4 = nn.utils.weight_norm(nn.Conv3d(96*factor, 128*factor, 3, padding=1))  # out: 8
        self.conv_4_1 = nn.utils.weight_norm(nn.Conv3d(128*factor, 128*factor, 3, padding=1))  # out: 8
        self.conv_5 = nn.utils.weight_norm(nn.Conv3d(128*factor, 128*factor, 3, padding=1))  # out: 8
        self.conv_5_1 = nn.utils.weight_norm(nn.Conv3d(128*factor, 128*factor, 3, padding=1))  # out: 8
        self.conv_6 = nn.utils.weight_norm(nn.Conv3d(128*factor, 128*factor, 3, padding=1))  # out: 8
        self.conv_6_1 = nn.utils.weight_norm(nn.Conv3d(128*factor, 128*factor, 3, padding=1))  # out: 8
        self.conv_7 = nn.utils.weight_norm(nn.Conv3d(128*factor, 128*factor, 3, padding=1))  # out: 8
        self.conv_7_1 = nn.utils.weight_norm(nn.Conv3d(128*factor, 128*factor, 3, padding=1))  # out: 8

        if res == 128:
            feature_size = (3 + input_dim + 32 + 64 + 96 + 128 + 128 + 128 + 128)
        else:
            feature_size = (3 + input_dim + 32*factor + 64*factor + 96*factor + 128*factor + 128*factor + 128*factor)
            
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_3 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_4 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, output_dim, 1))
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)
        
        if self.selfsup == True:
            self.fc_ss_0 = nn.utils.weight_norm(nn.Linear(128, hidden_dim))
            self.fc_ss_1 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim*2))
            self.fc_ss_2 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))
            self.fc_ss_3 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))
            self.fc_ss_4 = nn.utils.weight_norm(nn.Linear(hidden_dim*2, hidden_dim*2))            
            self.fc_ss_out = nn.utils.weight_norm(nn.Linear(hidden_dim*2, self.output_dim))  
            #self.fc_ss_out = nn.utils.weight_norm(nn.Linear(hidden_dim*2, 256))
            
        self.layers_first = [self.conv_1,
                        self.conv_2, self.conv_3, self.conv_4, 
                        self.conv_5, self.conv_7]
        
        self.layers_second = [self.conv_1_1,
                        self.conv_2_1, self.conv_3_1, self.conv_4_1, 
                        self.conv_5_1, self.conv_7_1]
                    
    def forward(self, x):
        features = [] 
        features.append(x)
        for i in range(len(self.layers_first)):
            x = self.actvn(self.layers_first[i](x))
            x = self.actvn(self.layers_second[i](x))
            features.append(x)
            if i<len(self.layers_first)-1:
                x = self.maxpool(x)

        self.features = [features[i] for i in range(len(features))]
                    
    def query(self, p):
        _B, _numpoints, _ = p.shape

        normalized_p = (p - self.b_min)/self.bb*2 - 1
        point_features = normalized_p.permute(0, 2, 1)
        for j, feat in enumerate(self.features):
            interpolation = F.grid_sample(feat, normalized_p.unsqueeze(1).unsqueeze(1), align_corners=False).squeeze(2).squeeze(2)
            point_features = torch.cat((point_features, interpolation), 1)

        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        point_features = self.actvn(self.fc_3(point_features))
        point_features = self.actvn(self.fc_4(point_features))
        point_features = self.fc_out(point_features)

        return point_features

    def self_sup(self):
        f1 = self.actvn(self.fc_ss_0(torch.squeeze(self.features[6])))
        f1 = self.actvn(self.fc_ss_1(f1))
        f1 = self.fc_ss_2(f1)
        f1 = self.fc_ss_3(f1)
        f1 = self.fc_ss_4(f1)
        f1 = self.fc_ss_out(f1)
        
        return f1




# class PointNetLVD(nn.Module):
#     def __init__(self, n_basis = 20,n_layers=4, size_layers=128, n_in = 3, normalize = False, feature_transform=False):
#         super(PointNetLVD, self).__init__()
#         self.k = n_basis
#         self.feature_transform = feature_transform
#         self.feat = PointNetfeat(n_in = n_in,n_layers=n_layers,size_layers=size_layers,normalize = False, global_feat=False, feature_transform=feature_transform)

#         hidden_dim=2048
#         output_dim=500*3
#         self.fc_0 = nn.utils.weight_norm(nn.Conv1d(256, hidden_dim, 1))
#         self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
#         self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
#         self.fc_3 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
#         self.fc_4 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
#         self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, output_dim, 1))
#         self.actvn = nn.ReLU()

#     def forward(self, x):
#         _B, _numpoints, _ = x.shape

#         normalized_p = x
#         point_features = normalized_p.permute(0, 2, 1)
#         point_features = torch.cat([point_features, self.feats], 1)

#         point_features = self.actvn(self.fc_0(point_features))
#         point_features = self.actvn(self.fc_1(point_features))
#         point_features = self.actvn(self.fc_2(point_features))
#         point_features = self.actvn(self.fc_3(point_features))
#         point_features = self.actvn(self.fc_4(point_features))
#         point_features = self.fc_out(point_features)

#         return point_features


#############
#############
#############


class IFNetPC(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=256, with_displacements=True):
        super(IFNetPC, self).__init__()
        self.with_displacements = with_displacements

        # 128**3 res input
        self.conv_in = nn.Conv3d(in_dim, 16, 3, padding=1, padding_mode="replicate")
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode="replicate")
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode="replicate")
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode="replicate")
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode="replicate")
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode="replicate")
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode="replicate")
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode="replicate")
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode="replicate")

        if self.with_displacements:
            feature_size = (in_dim + 16 + 32 + 64 + 128 + 128) * 7
        else:
            feature_size = in_dim + 16 + 32 + 64 + 128 + 128
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)

        if self.with_displacements:
            displacement = 0.0722
            displacements = []
            displacements.append([0, 0, 0])
            for x in range(3):
                for y in [-1, 1]:
                    input = [0, 0, 0]
                    input[x] = y * displacement
                    displacements.append(input)

            self.displacements = torch.Tensor(displacements)

    def forward(self, p, x):
        # x = x.unsqueeze(1)

        x_un = x.unsqueeze(1)

        p = p.unsqueeze(1).unsqueeze(1)

        if self.with_displacements:
            self.displacements = self.displacements.to(x_un.device)
            p = torch.cat([p + d for d in self.displacements], dim=2)  # (B,1,7,num_samples,3)

        feature_0 = F.grid_sample(
            x_un, p, padding_mode="border", align_corners=True
        )  # out : (B,C (of x), 1,1,sample_num)

        net = self.conv_in(x_un)
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(
            net, p, padding_mode="border", align_corners=True
        )  # out : (B,C (of x), 1,1,sample_num)

        # net_diffuse = net.clone()
        # for i, occ in zip(torch.arange(net.shape[0]),x):
        #     for j in torch.arange(net.shape[1]):
        #         for t in torch.arange(3 q):
        #             net_diffuse[i,j,:,:,:] = diffuse_heat(net_diffuse[i,j,:,:,:],occ,1)

        # #diffuse_heat(u_delta_t, occup, delta_t):

        # net = self.maxpool((net + net_diffuse)/2)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(
            net, p, padding_mode="border", align_corners=True
        )  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(
            net, p, padding_mode="border", align_corners=True
        )  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode="border", align_corners=True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode="border", align_corners=True)

        # here every channel corresponds to one feature
        features = torch.cat(
            (feature_0, feature_1, feature_2, feature_3, feature_4, feature_5), dim=1
        )  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(
            features, (shape[0], shape[1] * shape[3], shape[4])
        )  # (B, featues_per_sample, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out
