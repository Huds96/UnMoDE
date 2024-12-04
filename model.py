import torch
import torch.nn as nn
import torchvision.models as models
import math
import numpy as np
from torch.distributions import Normal, Independent, kl
from IVModule import Backbone


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet18FeatureExtractor, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-2])
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return features

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])
        self.adachannel = nn.Conv2d(2048, 512, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.adachannel(features)
        return features

class GaussianParameterPredictor(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(GaussianParameterPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.fc_mu = nn.Sequential(nn.Linear(input_dim, input_dim),
                                   nn.LayerNorm(input_dim),
                                   nn.Linear(input_dim, latent_dim)
                                   )
        self.fc_sigma = nn.Sequential(nn.Linear(input_dim, input_dim),
                                   nn.LayerNorm(input_dim),
                                   nn.Linear(input_dim, latent_dim)
                                   )
    
    def forward(self, x):
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        uncert = self.latent_dim / torch.sum((1.0 / sigma), dim=1)
        
        return mu, sigma, uncert

class GaussianParameterGaze(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(GaussianParameterGaze, self).__init__()
        self.latent_dim = latent_dim
        self.fc_mu = nn.Sequential(nn.Linear(input_dim, latent_dim//2),
                                   nn.LayerNorm(latent_dim//2),
                                   nn.Linear(latent_dim//2, latent_dim),
                                   )
        self.fc_sigma = nn.Sequential(nn.Linear(input_dim, latent_dim//2),
                                   nn.LayerNorm(latent_dim//2),
                                   nn.Linear(latent_dim//2, latent_dim),
                                   )
    
    def forward(self, x):
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        uncert = self.latent_dim / torch.sum((1.0 / torch.exp(sigma)), dim=1)
        
        return mu, sigma, uncert
    
class GaussianFeatureToMap(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(GaussianFeatureToMap, self).__init__()
        self.latent_dim = latent_dim
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim//2, 3),
            nn.BatchNorm2d(input_dim//2),
            nn.GELU(),
            nn.ConvTranspose2d(input_dim//2, input_dim//4, 3),
            nn.BatchNorm2d(input_dim//4),
            nn.GELU(),
            nn.ConvTranspose2d(input_dim//4, input_dim//8, 3),
            nn.BatchNorm2d(input_dim//8),
            nn.GELU(),
            nn.Conv2d(input_dim//8, input_dim, 1),
        )
        self.fc_refeature = nn.Sequential(nn.Linear(latent_dim, input_dim),
                                          nn.LayerNorm(input_dim),
                                          nn.Linear(input_dim, input_dim),
                                   )
    
    def forward(self, x):
        feature = self.fc_refeature(x)
        feature = feature.unsqueeze(-1).unsqueeze(-1)
        re_imagefeature = self.deconv(feature)
        
        return re_imagefeature


class GazeRegressor(nn.Module):
    def __init__(self, latent_dim, gaze_dim):
        super(GazeRegressor, self).__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, gaze_dim),)
    
    def forward(self, x):
        return self.fc(x)
    
class Emb2Feature(nn.Module):
    def __init__(self, latent_dim, featuredim):
        super(Emb2Feature, self).__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, latent_dim*2),
                                nn.LayerNorm(latent_dim*2),
                                nn.Linear(latent_dim*2, featuredim))
    
    def forward(self, x):
        return self.fc(x)

class conv1x1(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(conv1x1, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

        self.bn = nn.BatchNorm2d(out_planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

    def forward(self, feature):
        output = self.conv(feature)
        output = self.bn(output)
        output = self.avgpool(output)
        output = output.squeeze()
 
        return output

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResDeconv(nn.Module):
    def __init__(self, block, inplanes=2048):
        self.inplanes=inplanes
        super(ResDeconv, self).__init__()
        model = []
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 256, 2)] # 28
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 128, 2)] # 56
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 64, 2)] # 112
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 32, 2)] # 224
        model += [nn.Upsample(scale_factor=2)]
        model += [self._make_layer(block, 16, 2)] # 224
        model += [nn.Conv2d(16, 3, stride=1, kernel_size=1)]

        self.deconv = nn.Sequential(*model)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, features):
        img = self.deconv(features)
        return img

class Model(nn.Module):
    def __init__(self, latent_dim=128, gaze_dim=2, num_samples=64, inplanes=2048, channel=512):
        super(Model, self).__init__()
        transIn = 128
        convDims = [64, 128, 256, 512]
        self.id_feature_extractor = ResNet18FeatureExtractor()
        self.gaze_feature_extractor = Backbone(1, transIn, convDims)
        self.Decode = ResDeconv(BasicBlock, inplanes=inplanes)
        self.gaussian_predictor = GaussianParameterPredictor(latent_dim, latent_dim)
        self.ada = torch.nn.AdaptiveAvgPool2d(1)
        self.GazeEmb2Feature = GaussianFeatureToMap(latent_dim, channel)
        
        feature2embedding_list = []
        for i in range(len(convDims)):
            feature2embedding_list.append(GaussianParameterPredictor(latent_dim, latent_dim))
        self.feature2embedding_n = nn.ModuleList(feature2embedding_list)
        
        self.gaze_gaussian = GaussianParameterGaze(gaze_dim, latent_dim)
        self.gaze_regressor = GazeRegressor(latent_dim, gaze_dim)
        self.compressid = conv1x1(convDims[-1], latent_dim)
        gaze_regressor_list = []
        for i in range(len(convDims)):
            gaze_regressor_list.append(nn.Linear(latent_dim, gaze_dim))
        self.gaze_regressor_n = nn.ModuleList(gaze_regressor_list)
        
        self.gazefeatures_fusion = nn.Linear(latent_dim*(len(convDims)+1), latent_dim)
        
        self.num_samples = num_samples
        self.L1loss = torch.nn.L1Loss()
        self.MSE = torch.nn.MSELoss()
        self.fusion = nn.Conv2d(2*channel, inplanes, 1)
        self.featureloss = featureloss()
    
    def multiGazeSamples(self, features_list, nets, gazenets,label_dist, num_samples, batch_size, label):
        klloss_list = []
        samples_list = []
        gazes_list = []
        gazeloss_list = []
        for i, feature in enumerate(features_list):
            mu, sigma, uncert = nets[i](feature)
            gaze_dist = Independent(Normal(loc=mu, scale=torch.exp(sigma)),1)
            kl_div = kl.kl_divergence(gaze_dist, label_dist)
            kl_loss = kl_div.mean() / 1000
            klloss_list.append(kl_loss)
            sampled = gaze_dist.rsample([num_samples])
            gaze_samples = sampled.view(num_samples, -1)
            gaze_samples = torch.mean(gaze_samples, dim=0)
            gaze_samples = gaze_samples.view(batch_size, -1)
            gaze = gazenets[i](gaze_samples)
            gazeloss_list.append(self.L1loss(gaze, label))
            samples_list.append(gaze_samples)
            gazes_list.append(gaze)
        return klloss_list, samples_list, gazes_list, gazeloss_list
    
    def forward(self, x, label, train=False):
        batch_size = x.shape[0]
        gaze_features, gaze_features_list = self.gaze_feature_extractor(x)
        identity_featuremap = self.id_feature_extractor(x)
        identity_feature = self.compressid(identity_featuremap)
        
        floss = self.featureloss(gaze_features, identity_feature)
        
        mu, sigma, uncert = self.gaussian_predictor(gaze_features)
        label_mu, label_sigma, label_uncert = self.gaze_gaussian(label)
        face_dist = Independent(Normal(loc=mu, scale=torch.exp(sigma)),1)
        label_dist = Independent(Normal(loc=label_mu, scale=torch.exp(label_sigma)),1)
        
        kl_div = kl.kl_divergence(face_dist, label_dist)
        kl_loss = kl_div.mean() / 1000
        
        sampled = face_dist.rsample([self.num_samples])
        gaze_samples = sampled.view(self.num_samples, -1)
        gaze_samples = torch.mean(gaze_samples, dim=0)
        gaze_samples = gaze_samples.view(batch_size, -1)
        
        klloss_list, gaze_samples_list, gazes_list, gazeloss_list = self.multiGazeSamples(gaze_features_list, self.feature2embedding_n
                                                               , self.gaze_regressor_n, label_dist, self.num_samples, batch_size, label)
        
        
        gazefeatures_nlevel = torch.cat(gaze_samples_list, dim=1)
        gazefeatures_nlevel = torch.cat([gazefeatures_nlevel, gaze_samples], dim=1)
        gazefeatures_nlevel = self.gazefeatures_fusion(gazefeatures_nlevel)
        
        
        gaze_predictions = self.gaze_regressor(gaze_samples)
        gazeloss = self.L1loss(gaze_predictions, label)
        
        gloss = 2*gazeloss
        
        gaze_samples_feature = self.GazeEmb2Feature(gazefeatures_nlevel)
        identity_features = identity_featuremap
        deFeature = torch.cat((identity_features, gaze_samples_feature), dim=1)
        fusionFeature = self.fusion(deFeature)
        reImage = self.Decode(fusionFeature)
        reImage = torch.sigmoid(reImage)
        reloss = self.MSE(x, reImage)

        loss = gloss+floss+kl_loss+reloss
        return gaze_predictions, loss, [floss, kl_loss, gloss, reloss]

    def loss(self, x_in, label):
        label = label.normGaze
        gazes, loss, loss_list = self.forward(x_in["norm_face"], label)
        return loss, loss_list
    
class featureloss():
    def __init__(self):
        self.featureloss = torch.nn.CosineSimilarity()
        self.ada = torch.nn.AdaptiveAvgPool2d(1)

    def __call__(self, gaze, face):
        gaze1face1loss = self.featureloss(gaze, face)
        
        return (0.5*(1+gaze1face1loss)).mean()

