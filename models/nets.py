import torch.nn as nn

from models.utils import squeeze2d, unsqueeze2d, split2d, unsplit2d
from models.flows import ActNorm, Conv1x1, CouplingLayer


class Model(nn.Module):
    def __init__(self, num_levels, num_flows, conv_type, flow_type, num_blocks, hidden_channels, image_size=32,
                 in_channels=3):
        super(Model, self).__init__()
        self.num_levels = num_levels
        blocks = []
        for i in range(num_levels):
            in_channels = in_channels * 4
            image_size = image_size // 2
            flows = [Flow(conv_type, flow_type, num_blocks, in_channels, hidden_channels, image_size) for _ in
                     range(num_flows[i])]
            if i < num_levels - 1:
                in_channels = in_channels // 2
            blocks.append(Sequential(*flows))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, reverse=False, init=False):
        if not reverse:
            out = x
            outputs = []
            log_det_sum = x.new_zeros(x.size(0))
            for i in range(self.num_levels):
                out = squeeze2d(out)
                out, log_det = self.blocks[i](out, init=init)
                log_det_sum = log_det_sum + log_det
                if i < self.num_levels - 1:
                    out1, out2 = split2d(out, out.size(1) // 2)
                    outputs.append(out2)
                    out = out1
            out = unsqueeze2d(out)
            for _ in range(self.num_levels - 1):
                out2 = outputs.pop()
                out = unsqueeze2d(unsplit2d([out, out2]), factor=2)
        else:
            out = x
            outputs = []
            log_det_sum = x.new_zeros(x.size(0))
            out = squeeze2d(out)
            for _ in range(self.num_levels - 1):
                out1, out2 = split2d(out, out.size(1) // 2)
                outputs.append(out2)
                out = squeeze2d(out1)
            for i in reversed(range(self.num_levels)):
                if i < self.num_levels - 1:
                    out2 = outputs.pop()
                    out = unsplit2d([out, out2])
                out, log_det = self.blocks[i](out, reverse=reverse)
                log_det_sum = log_det_sum + log_det
                out = unsqueeze2d(out, factor=2)

        return out, log_det_sum


class Sequential(nn.Sequential):

    def forward(self, x, reverse=False, init=False):
        if not reverse:
            log_det_sum = x.new_zeros(x.size(0))
            for module in self._modules.values():
                x, log_det = module(x, init=init)
                log_det_sum = log_det_sum + log_det
        else:
            log_det_sum = x.new_zeros(x.size(0))
            for module in reversed(self._modules.values()):
                x, log_det = module(x, reverse=reverse)
                log_det_sum = log_det_sum + log_det

        return x, log_det_sum


class Flow(Sequential):

    def __init__(self, conv_type, flow_type, num_blocks, in_channels, hidden_channels, image_size):
        super(Flow, self).__init__()
        self.add_module('actnorm', ActNorm(in_channels, image_size))
        self.add_module('conv1x1', Conv1x1(in_channels, conv_type))
        self.add_module('couplinglayer', CouplingLayer(flow_type, num_blocks, in_channels, hidden_channels))
