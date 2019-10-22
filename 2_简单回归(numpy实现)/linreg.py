import numpy as np
import matplotlib.pyplot as plt

#预测目标y=w*x+b
#梯度更新
def step_grad(points,inital_b,inital_w,lr):
    b=inital_b
    w=inital_w
    grad_b=0
    grad_w=0
    for i in range(len(points)):
        x=points[i,0]
        y=points[i,1]
        grad_b+=2*(w*x+b-y)
        grad_w+=2*x*(w*x+b-y)
    grad_b/=float(len(points))
    grad_w/=float(len(points))

    b=b-lr*grad_b
    w=w-lr*grad_w
    return [w,b]

def gradient_reduce(points,inital_b,inital_w,lr,epoch):
    b=inital_b
    w=inital_w
    for i in range(epoch):
        w,b=step_grad(np.array(points),b,w,lr)
    return [w,b]

def calAvrError(w,b,points):
    error=0.0
    for i in range(len(points)):
        x=points[i,0]
        y=points[i,1]
        error+=(w*x+b-y)**2
    return error/float(len(points))

def plot_fit(points,w,b):
    points=np.array(points)
    max_x=np.max(points,axis=0)[0]
    min_x=np.min(points,axis=0)[0]

    plt.figure()
    plt.title('w:{:.2f},b:{:.2f},error:{:.2f}'.format(w,b,calAvrError(w,b,points)))
    plt.scatter(points[:,0],points[:,1],c='r',marker='o')
    plt.plot([min_x,max_x],[w*min_x+b,w*max_x+b],color='g')
    plt.show()


if __name__ == '__main__':
    data=np.genfromtxt('./data.csv',delimiter=',')
    inital_b, inital_w=0,0
    lr=0.0001
    epoch=1000
    w,b=gradient_reduce(data,inital_b,inital_w,lr,epoch)
    plot_fit(data,w,b)

