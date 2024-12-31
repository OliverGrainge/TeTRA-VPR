import torch 
import torch.nn as nn
import torch.nn.functional as F
import bitblas

@torch.no_grad()
def activation_quant_fake(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    dqx = (x * scale).round().clamp_(-128, 127) / scale
    return dqx, scale.squeeze()

@torch.no_grad()
def activation_quant_real(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    qx = (x * scale).round().clamp_(-128, 127).type(torch.int8)
    return qx, scale.squeeze()

@torch.no_grad()
def weight_quant_fake(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    dqw = (w * scale).round().clamp_(-1, 1) / scale
    return dqw, scale.squeeze()

@torch.no_grad()
def weight_quant_real(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    qw = (w * scale).round().clamp_(-1, 1).type(torch.int8)
    return qw, scale.squeeze()


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert torch.cuda.is_available(), "CUDA is not available"

        self.weight = nn.Parameter(torch.randn(out_features, in_features).cuda())
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features).cuda())
        else:
            self.register_parameter("bias", None)
        
        


        self.matmul_config = bitblas.MatmulConfig(
            M=1, 
            N=out_features, 
            K=in_features,
            A_dtype="int8", 
            W_dtype="int2", 
            accum_dtype="int32", 
            out_dtype="int32",
            layout="nt", 
            with_bias=False,
            group_size=None, 
            with_scaling=False, 
            with_zeros=False, 
            zeros_mode=None,
        )
        
        self.infer_matmul = bitblas.Matmul(config=self.matmul_config)
        qweight, scale = weight_quant_real(self.weight)
        self.register_buffer("qweight", self.infer_matmul.transform_weight(qweight))
        self.register_buffer("scale", scale)


    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.infer_forward(x)
    #
    def train_forward(self, x):
        dqx = x + (activation_quant_fake(x)[0] - x).detach()
        dqw = self.weight + (weight_quant_fake(self.weight)[0] - self.weight).detach()
        out = F.linear(dqx, dqw)
        if self.bias is not None: 
            out += self.bias.to(out.dtype)
        return out
    
    @torch.no_grad()
    def infer_forward(self, x):
        qx, act_scale = activation_quant_real(x)
        out = self.infer_matmul(qx, self.qweight) 
        out = out / act_scale / self.scale
        if self.bias is not None: 
           out += self.bias.to(out.dtype)
        return out

    @torch.no_grad()
    def eval(self):
        super().eval()
        qweight, scale = weight_quant_real(self.weight)
        self.weight.data = self.weight.data.cpu()
        if self.bias is not None: 
            self.bias.data = self.bias.data.cuda()
        self.register_buffer("qweight", self.infer_matmul.transform_weight(qweight).cuda())
        self.register_buffer("scale", scale.cuda())

    def train(self, mode=True): 
        super().train(mode)
        if mode:
            self._buffers.clear()
            self.weight.data = self.weight.data.cuda()
            if self.bias is not None: 
                self.bias.data = self.bias.data.cuda()


    def deploy(self, mode=True): 
        self.eval()
        del self.weight 

        


    


def test1(): # train cuda forward without bias
    layer = Linear(128, 256, bias=False)
    layer.train()
    x = torch.randn(1, 128).cuda()
    out = layer(x)


def test2():  # train cuda forward with bias
    layer = Linear(128, 256, bias=True)
    layer.train()
    x = torch.randn(1, 128).cuda()
    out = layer(x)

def test3(): # eval forwared without bias 
    layer = Linear(128, 256, bias=False)
    layer.eval()
    x = torch.randn(1, 128).cuda()
    out = layer(x)

def test4(): # eval forward with bias 
    layer = Linear(128, 256, bias=True)
    layer.eval()
    x = torch.randn(1, 128).cuda()
    out = layer(x)

def test5(): 
    layer = Linear(128, 256, bias=False)
    x = torch.randn(1, 128).cuda()
    layer.train()
    out_train = layer(x)
    layer.eval()
    out_eval = layer(x)
    
    assert out_train.dtype == out_eval.dtype
    assert torch.allclose(out_train, out_eval, rtol=2e-2, atol=2e-2)


def test6(): 
    layer = Linear(128, 256, bias=True)
    x = torch.randn(1, 128).cuda()
    layer.train()
    out_train = layer(x)
    layer.eval()
    out_eval = layer(x)
    print(out_eval.flatten()[:5])
    print(out_train.flatten()[:5])
    assert out_train.dtype == out_eval.dtype
    assert torch.allclose(out_train, out_eval, rtol=2e-2, atol=2e-2)


def test7(): 
    layer = Linear(128, 256, bias=True)
    layer.deploy()
    assert "weight" not in layer.state_dict().keys()
    assert "bias" in layer.state_dict().keys()
    assert "qweight" in layer.state_dict().keys()
    assert "scale" in layer.state_dict().keys()

def test8(): 
    layer = Linear(128, 256, bias=False)
    layer.deploy()
    print(layer.state_dict().keys())
    assert "weight" not in layer.state_dict().keys()
    assert "bias" not in layer.state_dict().keys()
    assert "qweight" in layer.state_dict().keys()
    assert "scale" in layer.state_dict().keys()


def test9():
    layer = Linear(128, 256, bias=False)
    layer.deploy()
    x = torch.randn(1, 128).cuda()
    out = layer(x)
    print(out.flatten()[:5])


def test10():
    layer = Linear(128, 256, bias=True)
    layer.deploy()
    x = torch.randn(1, 128).cuda()
    out = layer(x)
    print(out.flatten()[:5])





if __name__ == "__main__": 
    #test1()
    #test2()
    #test3()
    #test4()
    #test5()
    #test6()
    #test7()
    #test8()
    #test9()
    test10()