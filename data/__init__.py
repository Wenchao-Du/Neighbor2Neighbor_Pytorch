import numpy as np
from importlib import import_module
from torch.utils.data import Dataset


def get(args):
    data_name = args["data_name"]
    module_name = 'data.' + data_name.lower()
    module = import_module(module_name)

    return getattr(module, data_name)


class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)
