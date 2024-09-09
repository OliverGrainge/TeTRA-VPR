import numpy as np
from . import aggregators
from . import backbones


def get_backbone(backbone_arch='resnet50',
                 backbone_config={}):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where we sometimes need to crop the last residual block (ex. [4]). Defaults to [].

    Returns:
        model: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        if '18' in backbone_arch.lower():
            return backbones.ResNet(**backbone_config['resnet18'])
        elif '50' in backbone_arch.lower():
            return backbones.ResNet(**backbone_config['resnet50'])

    elif 'efficient' in backbone_arch.lower():
        return backbones.EfficientNet(**backbone_config['efficientnet'])

    elif 'swin' in backbone_arch.lower():
        return backbones.Swin(**backbone_config['swin'])
    
    elif 'dinov2' in backbone_arch.lower():
        return backbones.DINOv2(**backbone_config['dinov2'])
    


def get_aggregator(agg_arch='ConvAP', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        return aggregators.CosPlace(**agg_config['cosplace'])

    elif 'gem' in agg_arch.lower():
        return aggregators.GeMPool(**agg_config['gem'])
    
    elif 'convap' in agg_arch.lower():
        return aggregators.ConvAP(**agg_config['convap'])
    
    elif 'connected' in agg_arch.lower(): 
        return aggregators.FullyConnected(**agg_config['fully_connected'])
    

    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config['mixvpr']
        assert 'out_channels' in agg_config['mixvpr']
        assert 'in_h' in agg_config['mixvpr']
        assert 'in_w' in agg_config['mixvpr']
        assert 'mix_depth' in agg_config['mixvpr']
        return aggregators.MixVPR(**agg_config['mixvpr'])

    elif 'salad' in agg_arch.lower():
        assert 'num_channels' in agg_config['salad']
        assert 'num_clusters' in agg_config['salad']
        assert 'cluster_dim' in agg_config['salad']
        assert 'token_dim' in agg_config['salad']
        return aggregators.SALAD(**agg_config['salad'])
    






# -------------------------------------
def print_nb_params(m):
    """Prints the numbe of trainable parameters in the model

    Args:
        m (nn.Module): PyTorch model
    """
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')
    
    
def main():
    import torch
    
    x = torch.randn(1, 3, 224, 224) #random image
    # backbone = get_backbone(backbone_arch='resnet50')
    backbone = get_backbone(backbone_arch='resnet50')
    agg = get_aggregator('cosplace', {'in_dim':backbone.out_channels, 'out_dim':512})
    # agg = get_aggregator('GeM')
    print_nb_params(backbone)
    print_nb_params(agg)
    
    backbone_output = backbone(x)
    agg_output = agg(backbone_output)
    print(f'output shape: {agg_output.shape}')
    
if __name__ == '__main__':
    main()