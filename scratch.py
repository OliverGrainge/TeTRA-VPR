import torch
from pytorch_metric_learning import losses
import torch.nn.functional as F
from pytorch_metric_learning.distances import BaseDistance
import matplotlib.pyplot as plt
from pytorch_metric_learning.distances import BaseDistance

import torch
from pytorch_metric_learning import losses
import torch.nn.functional as F
from pytorch_metric_learning.distances import BaseDistance
import torch
from torch.autograd import Function

a = torch.randn(1024)

b = torch.tanh(a/0.05)
b_mask = b > 0
a_mask = a > 0
print((b_mask == a_mask).all())
plt.hist(a.detach().numpy(), label="unsquashed")
plt.hist(b.detach().numpy(), label="squashed")
plt.show()