import torch
from torch.nn import functional as F

x=torch.ones(1)
w=torch.full([1],2,requires_grad=True)
loss=F.mse_loss(torch.ones(1),x*w)

#method_1
#与方法二不要同时跑，因为backward后graph会消除，需要重建
loss.backward()
print(w.grad)

#method_2
w_grad=torch.autograd.grad(loss,[w])
print(w_grad)
