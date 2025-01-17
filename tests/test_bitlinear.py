import pytest
import torch 
import torch.nn as nn 
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.backbones.vitT import BitLinear, activation_quant_real, weight_quant_real

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
    x = torch.randn(1, input_features)
    out = layer(x)
    assert out.shape == (1, output_features)

@pytest.mark.cuda
def test_bitlinear_eval(input_features, output_features): 
    layer = BitLinear(input_features, output_features)
    layer.train()
    x = torch.randn(1, input_features)
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


