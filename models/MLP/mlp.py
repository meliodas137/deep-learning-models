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
    