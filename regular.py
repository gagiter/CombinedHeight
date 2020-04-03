import torch


def grad(x):
    conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    filter_x = torch.Tensor([[-0.25, 0.0, 0.25], [-0.5, 0.0, 0.5], [-0.25, 0.0, 0.25]])
    filter_x = filter_x.reshape(1, 1, 3, 3)
    filter_y = torch.Tensor([[-0.25, -0.5, -0.25], [0.0, 0.0, 0.0], [0.25, 0.5, 0.25]])
    filter_y = filter_y.reshape(1, 1, 3, 3)
    unbined = torch.unbind(x, dim=1)
    conv.weight = torch.nn.Parameter(filter_x, requires_grad=False)
    gdx = [conv(c.unsqueeze(1)) for c in unbined]
    gdx = torch.cat(gdx, dim=1)
    conv.weight = torch.nn.Parameter(filter_y, requires_grad=False)
    gdy = [conv(c.unsqueeze(1)) for c in unbined]
    gdy = torch.cat(gdy, dim=1)
    return gdx, gdy


def height_to_pos(height):
    lin_x = torch.linspace(0.0, 1.0, height.shape[-1], device=height.device)
    lin_y = torch.linspace(0.0, 1.0, height.shape[-2], device=height.device)
    grid_y, grid_x = torch.meshgrid([lin_x, lin_y])
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    grid = grid.repeat(height.shape[0], 1, 1, 1)
    pos = torch.cat([grid, height], dim=1)
    return pos


def planar(height):
    pos = height_to_pos(height)
    gdx, gdy = grad(height)
    n_z = torch.ones_like(height) / height.shape[-1]
    n = torch.cat([-gdx, -gdy, n_z], dim=1)
    norm = n.norm(p=2, dim=1, keepdim=True)
    n = n.div(norm.expand_as(n))
    d = n * pos
    d = -d.sum(dim=1, keepdim=True)
    return n, d


def overlap(height, label):
    ghx, ghy = grad(height)
    glx, gly = grad(label)
    glx = glx.abs().mean(dim=1, keepdim=True)
    gly = gly.abs().mean(dim=1, keepdim=True)
    o = ghx.abs() + ghy.abs()
    # o = ghx.abs() * torch.exp(-glx) + ghy.abs() * torch.exp(-gly)
    return o


def regular(height, label):
    n, _ = planar(height)
    o = overlap(height, label)
    return n, o


if __name__ == '__main__':
    height = torch.rand((1, 1, 512, 512))
    label = torch.rand((1, 3, 512, 512))
    n, o = regular(height, label)

