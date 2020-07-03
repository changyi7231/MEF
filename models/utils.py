import numpy as np
import torch

eps = 1e-8


def squeeze2d(x, factor=2):
    n, c, h, w = x.size()
    assert h % factor == 0 and w % factor == 0
    x = x.view(-1, c, h // factor, factor, w // factor, factor)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    x = x.view(-1, factor * factor * c, h // factor, w // factor)
    return x


def unsqueeze2d(x, factor=2):
    n, c, h, w = x.size()
    number = factor ** 2
    assert c >= number and c % number == 0
    x = x.view(-1, factor, factor, c // number, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
    x = x.view(-1, c // number, h * factor, w * factor)
    return x


def split2d(x, channels):
    z1 = x[:, :channels]
    z2 = x[:, channels:]
    return z1, z2


def unsplit2d(x):
    return torch.cat(x, dim=1)


def preprocess(image, bits, noise=None):
    bins = 2. ** bits
    image = image.mul(255)
    if bits < 8:
        image = torch.floor(image.div(256. / bins))
    if noise is not None:
        image = image + noise
    image = image.div(bins)
    image = (image - 0.5).div(0.5)
    return image


def postprocess(image, bits):
    bins = 2. ** bits
    image = image.mul(0.5) + 0.5
    image = image.mul(bins)
    image = torch.floor(image) * (256. / bins)
    image = image.clamp(0, 255).div(255)
    return image


def expm(x):
    """
    compute the matrix exponential: \sum_{k=0}^{\infty}\frac{x^{k}}{k!}
    """
    scale = int(np.ceil(np.log2(np.max([torch.norm(x, p=1, dim=-1).max().item(), 0.5]))) + 1)
    x = x / (2 ** scale)
    s = torch.eye(x.size(-1), device=x.device)
    t = x
    k = 2
    while torch.norm(t, p=1, dim=-1).max().item() > eps:
        s = s + t
        t = torch.matmul(x, t) / k
        k = k + 1
    for i in range(scale):
        s = torch.matmul(s, s)
    return s


def series(x):
    """
    compute the matrix series: \sum_{k=0}^{\infty}\frac{x^{k}}{(k+1)!}
    """
    s = torch.eye(x.size(-1), device=x.device)
    t = x / 2
    k = 3
    while torch.norm(t, p=1, dim=-1).max().item() > eps:
        s = s + t
        t = torch.matmul(x, t) / k
        k = k + 1
    return s
