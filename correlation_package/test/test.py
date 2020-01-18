import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import gradcheck
from correlation_package import Correlation 

import numpy as np

def test_correlation():
    A = Variable(torch.randn(2,3,100,100), requires_grad=True)
    A_ = A.cuda().float()
    B = Variable(torch.randn(2,3,100,100), requires_grad=True)
    B_ = B.cuda().float()

    model = Correlation(3, 3, 20, 1, 2, 1)
    y = model(A_, B_)
    
    print(y.size())

    print('Functional interface test passed')

    z = torch.mean(y)
    print('Forward output mean', z)

    z.backward()
    print(A.grad.size())
    print(B.grad.size())

    if A.grad is not None and B.grad is not None:
        print('Backward pass test passed')

    A = Variable(torch.randn(2,3,100,100), requires_grad=True)
    A_ = A.cuda().float()
    B = Variable(torch.randn(2,3,100,100), requires_grad=True)
    B_ = B.cuda().float()

    y = Correlation(3, 3, 20, 1, 2, 1)(A_, B_)
    print(y.size())

    print('Module interface test passed')

    z = torch.mean(y)
    z.backward()
    print(A.grad.size())
    print(B.grad.size())

    if A.grad is not None and B.grad is not None:
        print('Backward pass test passed')


if __name__=='__main__':
    test_correlation()
