import pytest
import torch 
import torch.nn as nn 
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbones.vitst import BitLinear, activation_quant_real, weight_quant_real

# Constants as fixtures
@pytest.fixture
def input_features():
    return 512

@pytest.fixture
def output_features():
    return 512

@pytest.mark.cuda
def test_bitlinear_train(input_features, output_features): 
    layer = BitLinear(input_features, output_features)
    layer.train()
    x = torch.randn(1, input_features).cuda()
    out = layer(x)
    assert out.shape == (1, output_features)

@pytest.mark.cuda
def test_bitlinear_eval(input_features, output_features): 
    layer = BitLinear(input_features, output_features)
    layer.train()
    x = torch.randn(1, input_features).cuda()
    out = layer(x)
    assert out.shape == (1, output_features)

@pytest.mark.cuda
def test_bitlinear_eval_train_equivalence(input_features, output_features): 
    layer = BitLinear(input_features, output_features)
    layer.train()
    layer = layer.cuda()
    x = torch.randn(1, input_features).cuda()
    out_train = layer(x)
    layer.eval()
    out_eval = layer(x)
    assert torch.allclose(out_train, out_eval, rtol=1e-3, atol=1e-3)

def test_sd_train_load(input_features, output_features): 
    layer = BitLinear(input_features, output_features)
    layer.train() 
    sd = layer.state_dict()
    layer = BitLinear(input_features, output_features)
    layer.load_state_dict(sd)

def test_sd_eval_load(input_features, output_features):
    layer = BitLinear(input_features, output_features)
    layer.eval()
    sd = layer.state_dict()
    layer = BitLinear(input_features, output_features)
    layer.eval()
    layer.load_state_dict(sd)

@pytest.mark.cuda
def test_Linear_BitLinear_equivalence(input_features, output_features):
    weight = torch.randint(-1, 2, (output_features, input_features)).cuda().float()
    layer = nn.Linear(input_features, output_features, bias=False)
    layer_bit = BitLinear(input_features, output_features, bias=False)

    for param in layer.parameters(): 
        param.requires_grad = False 

    for param in layer_bit.parameters(): 
        param.requires_grad = False 

    layer.eval()
    layer_bit.eval()
    layer = layer.cuda()
    layer_bit = layer_bit.cuda()

    layer.weight.data = weight.cuda()
    layer_bit.qweight.data = weight.to(torch.int8).cuda()
    layer_bit.scale.data = torch.tensor(1.0).cuda()

    input = torch.randint(-127, 127, (1, input_features)).cuda().float()
    out_fp = layer(input)
    out_bit = layer_bit(input)
    assert torch.mean(torch.abs(out_fp - out_bit)) < 1.0


