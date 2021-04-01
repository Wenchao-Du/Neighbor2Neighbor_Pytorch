#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   genmask.py
@Time    :   2021/03/30
@Author  :   Garified Du
@Version :   1.0
@Desc    :   if anything about the descriptions, please put them here. else None
'''

# here put the import lib

import torch
import torch.nn as nn
import torch.nn.functional as F
pattens = [
    torch.tensor([[1, 2], [0, 0]], dtype=torch.int32),
    # torch.tensor([[2, 1], [0, 0]], dtype=torch.int32),
    torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
    # torch.tensor([[0, 2], [0, 1]], dtype=torch.int32),
    # torch.tensor([[0, 0], [1, 2]], dtype=torch.int32),
    torch.tensor([[0, 0], [2, 1]], dtype=torch.int32),
    # torch.tensor([[1, 0], [2, 0]], dtype=torch.int32),
    torch.tensor([[2, 0], [1, 0]], dtype=torch.int32)
]
patten = torch.cat(pattens).view(4, 1, 2, 2)

op_pool = nn.MaxPool2d(2, 2)
""" test for single channel inputs
"""


def mask_generator(input):
    b, _, w, h = input.size()
    assert w % 2 == 0
    assert h % 2 == 0
    pattenmask = torch.randint(0, 4, (b, 1, w // 2, h // 2)).long()
    mask = torch.zeros((b, 1, w, h), dtype=torch.int32)
    for i in range(w // 2):
        for j in range(h // 2):
            mask[:, :, 2 * i:2 * i + 2,
                 2 * j:2 * j + 2] = patten[pattenmask[:, 0, i, j], :, :, :]
    """define maxpooling 2d operator replaceing the reshape operator
    """
    redmask = torch.where(mask == 1, torch.tensor(1), torch.tensor(0))
    print(op_pool(input * redmask))
    # print(input[bimask1].view(b, 1, w // 2, h // 2))
    bluemask = torch.where(mask == 2, torch.tensor(1), torch.tensor(0))
    print(op_pool(input * bluemask))
    # print(input[bimask2].view(b, 1, w // 2, h // 2))


""" class sampling for multi-channel inputs
"""


class masksampling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.maskpatten = torch.cat(pattens).view(4, 1, 2, 2)
        self.maskpatten = self.maskpatten.repeat(1, in_channels, 1, 1)
        # self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        b, _, w, h = input.size()
        assert w % 2 == 0
        assert h % 2 == 0
        """ generate random mask
        """
        initmask = torch.randint(0,
                                 4, (b, 1, w // 2, h // 2),
                                 dtype=torch.long)
        # initmask = initmask.repeat(1, c, 1, 1)
        fullmask = torch.zeros_like(input)
        """ need to optimize
        """
        for i in range(w // 2):
            for j in range(h // 2):
                fullmask[:, :, 2 * i:2 * i + 2, 2 * j:2 * j +
                         2] = self.maskpatten[initmask[:, 0, i, j], :, :, :]
        mask1 = torch.where(fullmask == 1.,
                            torch.tensor(1.).type_as(input),
                            torch.tensor(0.).type_as(input))
        mask2 = torch.where(fullmask == 2.,
                            torch.tensor(1.).type_as(input),
                            torch.tensor(0.).type_as(input))
        # redmask = self.pool(input * mask1)
        # bluemask = self.pool(input * mask2)

        return mask1, mask2


def generator(input, mask1, mask2):
    redinput = F.max_pool2d(input * mask1, kernel_size=2, stride=2)
    blueinput = F.max_pool2d(input * mask2, kernel_size=2, stride=2)
    return redinput, blueinput


class masksamplingv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(1, 2)

    def forward(self, input, patten):
        _, _, w, h = input.size()
        assert w % 2 == 0
        assert h % 2 == 0
        if patten == 0:
            output1 = self.pool(input)
            output2 = self.pool(input[:, :, 1:, :])
            return output2, output1
        elif patten == 1:
            output1 = self.pool(input)
            output2 = self.pool(input[:, :, 1:, :])
            return output2, output1
        elif patten == 2:
            output1 = self.pool(input)
            output3 = self.pool(input[:, :, :, 1:])
            return output1, output3
        elif patten == 3:
            output1 = self.pool(input)
            output3 = self.pool(input[:, :, :, 1:])
            return output3, output1
        elif patten == 4:
            output2 = self.pool(input[:, :, 1:, :])
            output4 = self.pool(input[:, :, 1:, 1:])
            return output4, output2
        elif patten == 5:
            output2 = self.pool(input[:, :, 1:, :])
            output4 = self.pool(input[:, :, 1:, 1:])
            return output2, output4
        elif patten == 6:
            output3 = self.pool(input[:, :, :, 1:])
            output4 = self.pool(input[:, :, 1:, 1:])
            return output3, output4
        elif patten == 7:
            output3 = self.pool(input[:, :, :, 1:])
            output4 = self.pool(input[:, :, 1:, 1:])
            return output4, output3
        else:
            raise Exception("no implemented patten")


if __name__ == "__main__":
    input = torch.rand((32, 3, 256, 256))
    model = masksamplingv2()
    for i in range(10):
        patten = torch.randint(0, 4, (1, ))
        print(patten.item())
        redinput, blueinput = model(input, patten.item())
        print(redinput)
        print(blueinput)
