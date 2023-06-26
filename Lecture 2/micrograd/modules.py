import numpy as np
import math
from abc import ABC,abstractmethod
import random
class Value:
    #when we create a single values Value object, we dont
    #pass in any children, however when we do an operation
    #we pass in the associated objects to be held as children
    def __init__(self,data,_children = (),_op = '',label = ''):
        #the input will be a tuple of children
        #but the data will be held as a Set
        #done for efficiency
        self.data = data
        #at initialization we assume that the value has no impact on loss
        self.grad = 0.0
        #a function which will do the chain rule at each node. 
        #how to chain upstream grad to local grad
        #for default, it doesnt do anything(e.g leaf node)
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
    def item(self):
        return self.data
    def __add__(self,other):
        #if other is already a Value object, leave it alone
        #else Value(other) means to wrap the values in a Value object
        other = other if isinstance(other,Value) else Value(other) 
        #here we are adding, meaning that a gradient is involved and an operation should be defined here
        out = Value(self.data + other.data,_children = (self,other),_op = '+')
        #we are assigning the backprop function to the "non leaf node"
        #these functions done return anything
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,_children = (self,other),_op = '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out 
    
    def __pow__(self,other):
        # other = other if isinstance(other,Value) else Value(other)
        assert isinstance(other,(int,float)),"Only accept int or float"
        out = Value(self.data ** other,_children = (self,),_op = f'**{other}')
        def _backward():
            self.grad += out.grad * (other*(self.data**(other-1)))
        out._backward = _backward
        return out
    
    def __rmul__(self,other):#other * self
        return self * other
    def __truediv__(self,other):
        #NOTE : If other is just a constant, 
        #__pow__ is not accessed
        #It is only accessed if other is a Value
        return self * (other**-1)
    def __rtruediv__(self,other):
        return other * (self**-1)
    
    def __radd__(self,other):
        return self + other

    def __neg__(self):
        #to implement negation, we mltiply by one. 
        #the backward pass for this will be done using the one in the
        #multiply method
        #NOTE: Only accessed if Self is a Value object
        return self * -1
    def __sub__(self,other):
        #the substract method is done by addition of a negation
        #the backward pass is implmented in the __add__ method
        return self + (-other)
    def __rsub__(self,other):
        return other + (-self)
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x),_children = (self,),_op = 'exp')
        def _backward():
            self.grad += out.grad*out.data
        out._backward = _backward
        return out
    def ln(self):
        x = self.data
        out = Value(np.log(x),_children = (self,),_op = 'ln')
        def _backward():
            self.grad += out.grad*(1/self.data)
        out._backward = _backward
        return out

    #we dont need to have the most atomic pieces of a function
    #we can have arbitary points of abstraction
    #only thing that matters is how to differentiate
    #create the local derivative
    def tanh(self):
        num = np.exp(self.data * 2) - 1
        den = np.exp(self.data * 2) + 1
        t = num/den
        out = Value(t,_children = (self,),_op='tanh')
        def _backward():
            self.grad += (1 - t**2)*out.grad
        out._backward = _backward
        return out
        # return c/d
    def ReLU(self):
        if self.data > 0:
            out = Value(self.data,_children = (self,),_op = 'ReLU')
        elif self.data <= 0:
            out = Value(0.0,_children = (self,),_op = 'ReLU')
        def _backward():
            if self.data > 0:
                self.grad += out.grad * 1
            elif self.data <= 0:
                self.grad += out.grad * 0
        out._backward = _backward
        return out
    def sigmoid(self):
        #1/1+exp(-z)
        z = -1 * self
        z = z.exp()
        z = 1 + z
        out = 1 * z**-1
        #an alternative
        # x = self.data
        # num = 1.0
        # dem = 1 + np.exp(-1 * x)
        # out = Value(num/dem,_children = (self,),_op='Sigmoid')
        # def _backward():
        #     self.grad += out.data(1 - out.data)
        return out
            
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            #maintain the visited nodes
            #This is important if one node is connected to multiple
            #other nodes
            if v not in visited:
                visited.add(v)
                #go through all the children nodes
                #and add them from right to left starting from
                #root node
                for child in v._prev:
                    build_topo(child)
                #Note that the first appending occurs at the leaf nodes
                #The initial root node adds itself to topo only after all the children
                #are added to it. 
                #This function guarantees that the parent is added only after the
                #children are added
                topo.append(v)
        build_topo(self)
        
        #reverse the topo from build_topo because it has the first 
        #entry as the leaf nodes
        self.grad = 1
        for node in reversed(topo):
            node._backward()



class Module(ABC):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    @abstractmethod
    def parameters(self):
        return []

class Neuron(Module):
    """
    A single neuron
    """
    def __init__(self,numin,act = 'ReLU'):
        """
        A single neuron with a single weight vector to be multiplied
        with a 
        numin : number of inputs coming in
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(numin)]
        self.b = Value(random.uniform(-1,1))
        assert act in ['tanh','ReLU','sigmoid','aaa'],'Invalid Activation Function'
        self.activation = act
    def __call__(self,x):
        #affine function = w.x + b
        #neuron activation
        act = sum((i*j for i,j in zip(self.w,x)),self.b)
        #passing through an activation function
        #act is a Value object
        activation = getattr(act,self.activation)
        out = activation()
        # if self.activation == 'ReLU':
        #     out = act.ReLU()
        # elif self.activation == 'tanh':
        #     out = act.tanh() 
        # elif self.activation == 'sigmoid':
        #     out = act.sigmoid()
        return out
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):
    """
    A Single layer of multiple neurons
    """
    def __init__(self,num_neurons_in,num_neurons_out):
        """
        num_neurons_in : the input dimensions coming into the Layer
        num_neurons_out : the number of outputs from that Layer
        At the initial layer, num_neurons_in will be the input dimension of 
        the training instance
        Initialize completely independent neurons with the num_neurons_in
        dimensionality.
        """
        self.neurons = [Neuron(num_neurons_in) for _ in range(num_neurons_out)]
    def __call__(self,x):
        #calculate the output for each neuron
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        #Identical to above
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params

class MLP(Module):
    def __init__(self,nin,nouts):
        #nin is the number of dimensions of a training instance (left most layer)
        #nouts is the number of neurons in each layer (This is expected to be a list)
        num_layers = 1 + len(nouts)
        num_inputs = [nin] + nouts
        self.layers = [Layer(num_inputs[i],num_inputs[i + 1]) for i in range(num_layers-1)]
    def __call__(self,x):
        #sequentially call each layer
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
