import torch
from torch import nn

class DatasetTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dataset):
        return dataset


class ImageDatasetTransform(DatasetTransform):
    def __init__(self, **kwargs):
        super().__init__()
        import kornia
        self.transform = torch.nn.Sequential(
            nn.ReplicationPad2d(4),
            kornia.augmentation.RandomCrop((84,84))
        #kornia.augmentation.RandomErasing(p=0.2),
        #kornia.augmentation.RandomAffine(degrees=0, translate=(2.0/84,2.0/84), p=1),
        #kornia.augmentation.RandomCrop((84,84))
    )

    def forward(self, dataset):
        dataset['obs'] = self.transform(dataset['obs'])
        return dataset