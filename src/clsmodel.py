import torch
import os

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from src.Classifier import model

root_path = 'FAX/src/Classifier/Logs'

model_urls = {
    'ffhqvanilla': os.path.join(root_path, 'FFHQVanilla/best.pth'),
    'ffhqresnet': os.path.join(root_path, 'FFHQResNet18/best.pth'),
    'ffhqdensenet': os.path.join(root_path, 'FFHQDenseNet121/best.pth'),

    'ffhqresnet-biased': os.path.join(root_path, 'FFHQResNet18-Biased/best.pth'),
    'ffhqdensenet-biased': os.path.join(root_path, 'FFHQDenseNet121-Biased/best.pth'),

    'afhqvanilla': os.path.join(root_path, 'AFHQVanilla/best.pth'),
    'afhqresnet': os.path.join(root_path, 'AFHQResNet18/best.pth'),
    'afhqdensenet': os.path.join(root_path, 'AFHQDenseNet121/best.pth'),

    'afhqresnet-biased': os.path.join(root_path, 'AFHQResNet18-Biased/best.pth'),
    'afhqdensenet-biased': os.path.join(root_path, 'AFHQDenseNet121-Biased/best.pth'),

    'mnistvanilla': os.path.join(root_path, 'MNISTVanilla/best.pth'),
    'mnistresnet': os.path.join(root_path, 'MNISTResNet18/best.pth'),
    'mnistdensenet': os.path.join(root_path, 'MNISTDenseNet121/best.pth'),

    'shapevanilla': os.path.join(root_path, 'SHAPESVanilla/best.pth'),
    'shaperesnet': os.path.join(root_path, 'SHAPESResNet18/best.pth'),
    'shapedensenet': os.path.join(root_path, 'SHAPESDenseNet121/best.pth'),

}



def afhq(pretrained=None, feature_extractor='densenet', case = 'fair'):
    if feature_extractor.lower().__contains__('vanilla'):
        return afhqvanilla(32, True)
    
    nclass = 3
    if feature_extractor.lower().__contains__('densenet'):
        net = model.DenseNet121(num_channel=3, classCount=nclass)
        url_key = 'afhqdensenet'
    else:
        net = model.ResNet18(num_channel=3, classCount=nclass)
        url_key = 'afhqresnet'

    if case == 'biased': url_key = url_key+'-biased'
    model_url = model_urls[url_key]
    net.num_classes = nclass

    if (pretrained != None) and (case != 'random'):
        m = torch.load(model_url)
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)

        print (f"AFHQ-{feature_extractor}-{case}: {model_url} weights loaded ............")
    return net


def ffhq(pretrained=None, feature_extractor='densenet', case = 'fair'):
    if feature_extractor.lower().__contains__('vanilla'):
        return ffhqvanilla(32, True)
    
    nclass = 2
    if feature_extractor.lower().__contains__('densenet'):
        net = model.DenseNet121(num_channel=3, classCount=nclass)
        url_key = 'ffhqdensenet'
    else:
        net = model.ResNet18(num_channel=3, classCount=nclass)
        url_key = 'ffhqresnet'

    if case == 'biased': url_key = url_key+'-biased'
    model_url = model_urls[url_key]
    net.num_classes = nclass

    if (pretrained != None) and (case != 'random'):
        m = torch.load(model_url)
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)

        print (f"FFHQ-{feature_extractor}-{case} weights loaded ............")
    return net


def mnist(pretrained=None, feature_extractor='densenet', case = 'fair'):
    if feature_extractor.lower().__contains__('vanilla'):
        return mnistvanilla(32, True)
    
    nclass = 10
    if feature_extractor.lower().__contains__('densenet'):
        net = model.DenseNet121(num_channel=3, classCount=nclass)
        model_url = model_urls['mnistdensenet']
    else:
        net = model.ResNet18(num_channel=3, classCount=nclass)
        model_url = model_urls['mnistresnet']
    net.num_classes = nclass

    if (pretrained != None) and (case != 'random'):
        m = torch.load(model_url)
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)

        print (f"MNIST-{feature_extractor}-{case}  weights loaded ............")
    return net



def shapes(pretrained=None, feature_extractor='densenet', case = 'fair'):
    if feature_extractor.lower().__contains__('vanilla'):
        return shapesvanilla(32, True)
    
    nclass = 4
    if feature_extractor.lower().__contains__('densenet'):
        net = model.DenseNet121(num_channel=3, classCount=nclass)
        model_url = model_urls['shapedensenet']
    else:
        net = model.ResNet18(num_channel=3, classCount=nclass)
        model_url = model_urls['shaperesnet']
    net.num_classes = nclass

    if (pretrained != None) and (case != 'random'):
        m = torch.load(model_url)
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)

        print (f"SHAPE-{feature_extractor}-{case}  weights loaded ............")
    return net


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)


def mnistvanilla(n_channel, pretrained=None):
    cfg = [
        # n_channel, 'M',
        # 2*n_channel, 'M',
        # 4*n_channel, 'M',
        # 4*n_channel, 'M',
        n_channel, 'M', 
        n_channel, 'M', 
        2*n_channel, 'M',
    ]

    layers = make_layers(cfg, batch_norm=True)
    net = model.SVHN(layers, n_channel=2*n_channel, num_classes=10)
    if pretrained != None:
        m = torch.load(model_urls['mnistvanilla'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)
    return net


def afhqvanilla(n_channel, pretrained=None):
    cfg = [
        # n_channel, 'M',
        # 2*n_channel, 'M',
        # 4*n_channel, 'M',
        # 4*n_channel, 'M',
        n_channel, 'M', 
        n_channel, 'M', 
        2*n_channel, 'M',
    ]
    
    
    layers = make_layers(cfg, batch_norm=True)
    net = model.SVHN(layers, n_channel=2*n_channel, num_classes=3)
    if pretrained != None:
        m = torch.load(model_urls['afhqvanilla'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)
    return net




def ffhqvanilla(n_channel, pretrained=None):
    cfg = [
        # n_channel, 'M',
        # 2*n_channel, 'M',
        # 4*n_channel, 'M',
        # 4*n_channel, 'M',
        n_channel, 'M', 
        n_channel, 'M', 
        2*n_channel, 'M',
    ]
    
    layers = make_layers(cfg, batch_norm=True)
    net = model.SVHN(layers, n_channel=32*n_channel, num_classes=2)
    
    # net = nn.Sequential(net.features,
    #                     nn.Flatten(),
    #                     net.classifier)

    if pretrained != None:
        m = torch.load(model_urls['ffhqvanilla'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)
    
    return net


def shapesvanilla(n_channel, pretrained=None):
    cfg = [
        # n_channel, 'M',
        # 2*n_channel, 'M',
        # 4*n_channel, 'M',
        # 4*n_channel, 'M',
        n_channel, 'M', 
        n_channel, 'M', 
        2*n_channel, 'M',
    ]


    layers = make_layers(cfg, batch_norm=True)
    net = model.SVHN(layers, n_channel=2*n_channel, num_classes=4)
    if pretrained != None:
        m = torch.load(model_urls['shapevanilla'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        net.load_state_dict(state_dict)
    return net