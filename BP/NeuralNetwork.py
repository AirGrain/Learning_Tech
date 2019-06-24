import numpy as np
class NeuralNetwork(object):
    def __init__(self, layers = [2 , 10, 1], activations=['sigmoid', 'sigmoid']): 
        assert(len(layers) == len(activations)+1) 
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1): 
            self.weights.append(np.random.randn(layers[i+1], layers[i])) 
            self.biases.append(np.random.randn(layers[i+1], 1)) 
    
    def feedforward(self, x):  # 正向计算
        #x 在train函数里为x_batch，x，y是一个矩阵：相当于对多笔数据进行并行计算
        a = np.copy(x) 
        z_s = [] 
        a_s = [a]
        for i in range(len(self.weights)): 
            activation_function = self.getActivationFunction(self.activations[i]) 
            z_s.append(self.weights[i].dot(a) + self.biases[i]) 
            a = activation_function(z_s[-1]) 
            a_s.append(a)
        return (z_s, a_s)
    
    def backpropagation(self,y, z_s, a_s): # 反向计算
            dw = []  # dl/dW
            db = []  # dl/dB
            deltas = [None] * len(self.weights)  # 存放每一层的error
            # deltas[-1] = sigmoid'(z)*[partial l/partial y]
            # 这里y是标注数据，a_s[-1]是最后一层的输出，差值就是二范数loss的求导
            deltas[-1] =(y-a_s[-1])*(self.getDerivitiveActivationFunction(self.activations[-1]))(z_s[-1])
            # Perform BackPropagation 
            for i in reversed(range(len(deltas)-1)): 
                deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.getDerivitiveActivationFunction(self.activations[i])(z_s[i]))
            batch_size = y.shape[1]
            db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
            dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
            # return the derivitives respect to weight matrix and biases
            return dw, db

    def train(self, x, y, batch_size=5, epochs=100, lr = 0.01):
        # update weights and biases based on the output
        for e in range(epochs):
            '''
            # 使用下标来打乱数据,有点麻烦
            x_num = x.shape[0] 
            index = np.arange(x_num)  # 生成下标  
            np.random.shuffle(index)  
            i = index[0]
            '''
            # 直接打乱源数据
            nn=np.random.randint(1,1000)
            np.random.seed(nn) 
            np.random.shuffle(x) 
            np.random.seed(nn) 
            np.random.shuffle(y) 
            i = 0
            while(i<len(y)):
                x_batch = x[i:i+batch_size].reshape(1, -1) # 转换成矩阵更加清晰明了
                y_batch = y[i:i+batch_size].reshape(1, -1) 
                i = i+batch_size
                z_s, a_s = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s) 
                # 一个batch更新一次参数
                self.weights = [w+lr*dweight for w,dweight in  zip(self.weights, dw)]
                self.biases = [w+lr*dbias for w,dbias in  zip(self.biases, db)]
            print("loss = {}".format(np.linalg.norm(a_s[-1]-y_batch) ))

    @staticmethod
    def getActivationFunction(name):
        if(name == 'sigmoid'):
            return lambda x : 1.0/(1+np.exp(-x))
        elif(name == 'tanh'):
            return lambda x : np.tanh(x)
        elif(name == 'relu'):
            def relu(x):
                y = np.copy(x)
                y[y<0] = 0
                return y
            return relu
        else:
            print('Unknown activation function. linear is used')
            return lambda x: x
    
    @staticmethod
    def getDerivitiveActivationFunction(name):
        if(name == 'sigmoid'):
            sig = lambda x : 1/(1+np.exp(-x))
            return lambda x :sig(x)*(1-sig(x)) 
        elif(name == 'tanh'):
            return lambda x: 1.0 - np.tanh(x)**2
        elif(name == 'relu'):
            def relu_diff(x):
                y = np.copy(x)
                y[y>=0] = 1
                y[y<0] = 0
                return y
            return relu_diff
        else:
            print('Unknown activation function. linear is used')
            return lambda x: 1

if __name__=='__main__':
    import matplotlib.pyplot as plt
    nn = NeuralNetwork([1,100,80,1],activations=['sigmoid','tanh','tanh'])
    X = 2*np.pi*np.random.rand(1000)
    y = np.sin(X)
    # lr设置太小，epoch的值则应该比较大，但也不能太大，否则大部分计算会浪费了，一直在局部极小值跳来跳去，还是自适应lr的优化算法比较好
    nn.train(X, y, epochs=1000, batch_size=10, lr = 0.01)
    
    xx=X.reshape(1, -1)  
    yy=y.reshape(1, -1)
    _, a_s = nn.feedforward(xx) # 使用新模型
    plt.scatter(xx.flatten(), yy.flatten(),color='blue')
    plt.scatter(xx.flatten(), a_s[-1].flatten(),color='red')
    plt.show()
