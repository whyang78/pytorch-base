import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import Axes3D

#优化实例：f(x,y)= (x**2+y-11)**2 +(x+y**2-7)**2

def result(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z=result([X,Y])

#绘制3D图像
fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#参数
#初始点不一样，最后到达的点也不一样
x=torch.tensor([0.,0.],requires_grad=True)
optimizer=optim.Adam([x],lr=0.001)
for step in range(20000):
    pred=result(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()
    if step%2000==0:
        print('step:{},x:{},result:{}'.format(step,x.tolist(),pred.item()))
