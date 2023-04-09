import torch
import torch.nn as nn
import numpy as np
class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        f = self.getFunction(self.f_function)
        g = self.getFunction(self.g_function)
        y_hat = torch.zeros(x.size()[0], self.parameters['W2'].size()[0])
        h_cache = torch.zeros(x.size()[0], self.parameters['W1'].size()[0])
        for i, input in enumerate(x):
            s1 = torch.matmul(self.parameters['W1'], input) + self.parameters['b1']
            h = f(s1)
            s2 = torch.matmul(self.parameters['W2'], h) + self.parameters['b2']
            o = g(s2)
            y_hat[i] = o
            h_cache[i] = h

        self.cache['input'] = x
        self.cache['h'] = h_cache
        self.cache['y_hat'] = y_hat
        return y_hat
            

    
    def backward(self, dJdy_hat):
        for idx, error in enumerate(dJdy_hat):
            self.back_prop(error, idx, "second")
            self.back_prop(error, idx, "first")
        return


    def back_prop(self, error, idx, layer):
        # pass
        if layer == "second":
            derivative = self.getDerivative(self.g_function, self.cache['y_hat'][idx])
            a = torch.reshape(self.cache['h'][idx], (self.cache['h'][idx].size()[0], 1))
            b = torch.reshape(error * derivative, (1, derivative.size()[0]))
            self.cache['b'] = error * derivative
            self.grads['dJdb2'] += error * derivative
            self.grads['dJdW2'] += torch.transpose(torch.matmul(a, b), 0, -1)

        elif layer == "first":
            derivative = self.getDerivative(self.f_function, self.cache['h'][idx])
            a = torch.reshape(self.cache['input'][idx], (self.cache['input'][idx].size()[0], 1))
            temp = torch.matmul(self.cache['b'], self.parameters['W2'])
            self.grads['dJdb1'] += temp * derivative
            b = torch.reshape( temp * derivative, (1, derivative.size()[0]))
            self.grads['dJdW1'] += torch.transpose(torch.matmul(a, b), 0, -1)

    def getDerivative(self, function, x):
        if function == "relu":
            return (x > 0)*1
        elif function == "sigmoid":
            return x*(1-x)
        return torch.ones(x.size()[0])

    def getFunction(self, function):
        if function == "relu":
            return nn.ReLU()
        elif function == "sigmoid":
            return nn.Sigmoid()
        return nn.Identity()
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()
