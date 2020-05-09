import  random
import  numpy as np

np.random.seed(78)
random.seed(78)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))

class MLP_NP:
    def __init__(self,sizes):
        self.sizes=sizes
        self.num_layers=len(self.sizes)-1

        self.biases=[np.random.randn(ch,1) for ch in self.sizes[1:]]
        self.weights=[np.random.randn(ch_out,ch_in) for ch_in,ch_out in zip(self.sizes[:-1],self.sizes[1:])]

    def forward(self,x):
        for i in range(self.num_layers):
            x=np.dot(self.weights[i],x)+self.biases[i]
            x=sigmoid(x)
        return x

    def backprop(self,x,y):
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]

        # 1.forward
        zs=[]
        activations=[x]
        activation=x
        for w,b in zip(self.weights,self.biases):
            z=np.dot(w,activation)+b
            activation=sigmoid(z)

            zs.append(z)
            activations.append(activation)
        loss=np.power(activations[-1]-y,2).sum()

        # 2. backward
        delta=activations[-1]*(1-activations[-1])*(activations[-1]-y)
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-1-1].T)

        for l in range(2,self.num_layers+1):
            l=-l
            z=zs[l]
            a=activations[l]

            delta=np.dot(self.weights[l+1].T,delta)*a*(1-a)
            nabla_b[l]=delta
            nabla_w[l]=np.dot(delta,activations[l-1].T)
        return nabla_w,nabla_b,loss

    def train(self,training_data,epochs,batchsz,lr,test_data):
        if test_data:
            n_test=len(test_data)

        n=len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches=[
                training_data[k:k+batchsz]
                for k in range(0,n,batchsz)
            ]

            total_loss=0
            for mini_batch in mini_batches:
                loss=self.update_mini_batch(mini_batch,lr)
                total_loss+=loss
            total_loss=total_loss/n
            if test_data:
                print('Epoch {}: {} / {} , loss:{:.6f}'.format(i,self.evaluate(test_data),n_test,total_loss))

            else:
                print('Epoch {} complete'.format(i))

    def update_mini_batch(self,batch,lr):
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        nabla_b=[np.zeros(b.shape) for b in self.biases]

        total_loss=0
        for x,y in batch:
            nabla_w_,nabla_b_,loss=self.backprop(x,y)
            nabla_w=[accu+cur for accu,cur in zip(nabla_w,nabla_w_)]
            nabla_b=[accu+cur for accu,cur in zip(nabla_b,nabla_b_)]
            total_loss+=loss

        nabla_w=[w/len(batch) for w in nabla_w]
        nabla_b=[b/len(batch) for b in nabla_b]

        self.weights=[w-lr*nabla for w,nabla in zip(self.weights,nabla_w)]
        self.biases=[b-lr*nabla for b,nabla in zip(self.biases,nabla_b)]
        return total_loss

    def evaluate(self,test_data):
        result=[(np.argmax(self.forward(x)),y) for x,y in test_data]
        correct=sum([int(pred==y) for pred,y in result])
        return correct

def main():
    import mnist_loader
    training_data, validation_data, test_data= mnist_loader.load_data_wrapper()

    model = MLP_NP([784, 30, 10])
    model.train(training_data, 1000, 10, 0.1, test_data)

if __name__ == '__main__':
    main()