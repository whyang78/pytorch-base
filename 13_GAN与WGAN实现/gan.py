import torch
from torch import nn, optim, autograd
import numpy as np
import visdom
from torch.nn import functional as F
from matplotlib import pyplot as plt
import random

h_dim = 400
batchsz = 512
viz = visdom.Visdom()

#生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self, z):
        output = self.net(z)
        return output

#判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.normal_(0.0, 0.02)
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        output = self.net(x)
        return output

#数据集构建
def data_generator():
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(batchsz):
            point = np.random.randn(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev
        yield dataset

def d_loss(logits_real,logits_fake):
    real_labels=torch.ones((logits_real.size(0),1),dtype=torch.float32,device=logits_real.device)
    fake_labels=torch.zeros((logits_fake.size(0),1),dtype=torch.float32,device=logits_fake.device)
    loss=F.binary_cross_entropy_with_logits(logits_real,real_labels)+\
         F.binary_cross_entropy_with_logits(logits_fake,fake_labels)
    return loss

def g_loss(logits_fake):
    real_labels=torch.ones((logits_fake.size(0),1),dtype=torch.float32,device=logits_fake.device)
    loss=F.binary_cross_entropy_with_logits(logits_fake,real_labels)
    return loss

def gradient_penalty(D, xr, xf):
    LAMBDA = 0.3

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, 2]
    alpha = torch.rand(batchsz, 1).cuda()
    alpha = alpha.expand_as(xr)

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gp

def main():
    torch.manual_seed(23)
    np.random.seed(23)

    G = Generator().cuda()
    D = Discriminator().cuda()

    optim_G = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

    data_iter = data_generator()

    viz.line([[0,0]], [0], win='loss', opts=dict(title='loss',legend=['D', 'G']))

    for epoch in range(50000):
        # 1. train discriminator for k steps
        for _ in range(5):
            x = next(data_iter)
            xr = torch.from_numpy(x).cuda()
            logits_real = D(xr)

            z = torch.randn(batchsz, 2).cuda()
            xf = G(z).detach()
            logits_fake = D(xf)

            # gradient penalty
            gp = gradient_penalty(D, xr, xf)

            loss_D = d_loss(logits_real,logits_fake)+gp
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batchsz, 2).cuda()
        xf = G(z)
        logits_fake = D(xf)

        loss_G = g_loss(logits_fake)
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch+1], win='loss', update='append')

if __name__ == '__main__':
    main()