from utils import *
import torch
import matplotlib.pyplot as plt

def f(x):
    return LinearQuantizerHardTanh.apply(x, 4, 0, 6)


x = torch.arange(-10, 10, 0.01)
x.requires_grad = True
y = f(x)

y.backward(torch.ones_like(y))

plt.figure()
plt.plot(x.detach().numpy(), y.detach().numpy())
plt.plot(x.detach().numpy(), x.grad.detach().numpy())

plt.show()

#print(x.grad)