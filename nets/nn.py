import math

import torch

from utils.util import make_anchors, initialize_weights


def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] 
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2)
        self.conv2 = Conv(in_ch, out_ch // 2)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))

class Bottleneck(torch.nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(torch.nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = torch.nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        p1 = [Conv(width[0], width[1], 3, 2)]
        p2 = [Conv(width[1], width[2], 3, 2),
              C2f(width[2], width[2], depth[0], True)]
        p3 = [Conv(width[2], width[3], 3, 2),
              C2f(width[3], width[3], depth[1], True)]
        p4 = [Conv(width[3], width[4], 3, 2),
              C2f(width[4], width[4], depth[2], True)]
        p5 = [Conv(width[4], width[5], 3, 2),
              C2f(width[5], width[5], depth[0], True),
              SPP(width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = C2f(width[4] + width[5], width[4], depth[0], False)
        self.h2 = C2f(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = C2f(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = C2f(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)

class SAVPE(torch.nn.Module):
    def __init__(self, filters, mid_ch, embed_dims):
        super().__init__()

        self.sematic = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, mid_ch, 3), Conv(mid_ch, mid_ch, 3)) for x in filters)
        self.sematic[1].append(torch.nn.Upsample(scale_factor=2))
        self.sematic[2].append(torch.nn.Upsample(scale_factor=4))
        
        self.activation = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, mid_ch, 1)) for x in filters)
        self.activation[1].append(torch.nn.Upsample(scale_factor=2))
        self.activation[2].append(torch.nn.Upsample(scale_factor=4))
        
        self.c = 16
        self.cv3 = torch.nn.Conv2d(3 * mid_ch, embed_dims, 1)
        self.cv4 = torch.nn.Conv2d(3 * mid_ch, self.c, 3, padding=1)
        self.cv5 = torch.nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = torch.nn.Sequential(Conv(2 * self.c, self.c, 3), torch.nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x, vp):
        y = [self.activation[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))
        
        x = [self.sematic[i](xi) for i, xi in enumerate(x)]
        
        x = self.cv3(torch.cat(x, dim=1))
        
        B, C, H, W = x.shape 

        Q = vp.shape[1]
        
        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)
        
        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))
        
        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
 
        score = torch.nn.functional.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)
        
        return torch.nn.functional.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)

class BNContrastiveHead(torch.nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = torch.nn.Parameter(torch.tensor([-2.0]))
        # use -1.0 is more stable
        self.logit_scale = torch.nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        # w = F.normalize(w, dim=-1, p=2)
        
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias

class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, filters=(), embed_dims=512):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = 1
        self.nl = len(filters)  # number of detection layers
        self.no = self.nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.embed_dims = embed_dims

        # c1 = max(filters[0], self.nc) MID_CH
        # c2 = max((filters[0] // 4, self.ch * 4)) BOX

        box = max(64, filters[0] // 4)
        mid_ch = max(80, filters[0])
        self.mid_ch = mid_ch

        self.dfl = DFL(self.ch)
       
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box, 3),
                                                           Conv(box, box, 3),
                                                           torch.nn.Conv2d(box, 4 * self.ch, 1)) for x in filters)
        

        self.emb = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, mid_ch, 3),
                                                           Conv(mid_ch, mid_ch, 3),
                                                           torch.nn.Conv2d(mid_ch, embed_dims, 1)) for x in filters)
        
        self.savpe = SAVPE(filters, mid_ch, embed_dims)
        self.bn = torch.nn.ModuleList(BNContrastiveHead(embed_dims) for _ in filters)
    
    def get_vpe(self, prompt, prompt_mask):
        return self.savpe(prompt, prompt_mask)

    def forward(self, x, vpe):
        for i, (box, emb, bn) in enumerate(zip(self.box, self.emb, self.bn)):
            x[i] = torch.cat(tensors=(box(x[i]), bn(emb(x[i]) , vpe)), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no , -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        m = self
        for a, b, s in zip(m.box, m.emb, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.mid_ch / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, embed_dims=512):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        vpe_dummy = torch.zeros(1, 1, embed_dims)

        self.head = Head((width[3], width[4], width[5]), embed_dims)
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy, vpe_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()
        initialize_weights(self)

    def forward(self, x, vpe):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x), vpe)
    
    def get_vpe(self, p, p_mask):
        p = self.net(p)
        p = self.fpn(p)
        return self.head.get_vpe(p, p_mask)
    
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v8_n():
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth)


def yolo_v8_s():
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth)


def yolo_v8_m():
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth)


def yolo_v8_l():
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth)


def yolo_v8_x():
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth)

def load_model(model_path):
    model = yolo_v8_s()
    weights = torch.load(model_path, weights_only=False)
    pretrained_state_dict = weights.state_dict()
    # new_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('head.cls')}

    model.load_state_dict(pretrained_state_dict, strict=False)
    return model