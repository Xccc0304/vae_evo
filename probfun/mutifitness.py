import torch
import math



def zdt1(x):
    x = x.clamp(1e-6, 1 - 1e-6)
    f1 = x[:, 0]
    g = 1.0 + 9.0 * x[:, 1:].mean(dim=1)
    ratio = (f1 / (g + 1e-6)).clamp(0.0, 1.0)
    f2 = g * (1.0 - torch.sqrt(ratio))
    return f1, f2

def zdt2(x):

    x = torch.clamp(x, 0.0, 1.0)
    f1 = x[:, 0]
    g = 1.0 + 9.0 * x[:, 1:].mean(dim=1)
    f1_over_g = f1 / (g + 1e-6)
    f2 = g * (1.0 - f1_over_g ** 2)

    return f1, f2


def zdt3(x):
    x = x.clamp(1e-8, 1.0)
    f1 = x[:, 0]
    g = 1 + 9 * torch.mean(x[:, 1:], dim=1)
    h = 1 - torch.sqrt(f1 / (g + 1e-8)) - f1 / (g + 1e-8) * torch.sin(10 * torch.pi * f1)
    f2 = g * h
    return f1, f2




def zdt_beta(x):
    x = x.clamp(1e-6, 1 - 1e-6)
    f1 = x[:, 0]
    n = x.size(1)
    g = 1.0 + 9.0 * x[:, 1:].sum(dim=1) / (n - 1)
    ratio = (f1 / (g + 1e-8)).clamp(0.0, 1.0)
    f2 = g * (1.0 - (ratio + 1e-8) ** 0.25)
    return f1, f2


def fon(x):

    n = x.shape[1]
    inv_sqrt_n = 1.0 / torch.sqrt(torch.tensor(n))

    f1 = 1.0 - torch.exp(-torch.sum((x - inv_sqrt_n)**2, dim=1))
    f2 = 1.0 - torch.exp(-torch.sum((x + inv_sqrt_n)**2, dim=1))

    return f1, f2


def mt1(x):
    x = x.clamp(0.0, 1.0)
    x1 = x[:, 0]
    x2 = x[:, 1]

    f1 = x1
    f2 = 1 - x1**2 +  (x2 - 0.5)**2
    return f1, f2


def sch(x):

    f1 = x[:, 0] ** 2
    f2 = (x[:, 0] - 2) ** 2
    return f1, f2









