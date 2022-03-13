import numpy as np
from celeba import CelebADataset


class CelebaCustomDataset_V2(CelebADataset):
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        target = target['attributes'] == 1
        new_target = target[20]
        return image, new_target


class CelebaCustomDatasetRef_V2(CelebADataset):
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        target = target['attributes'] == 1
        new_target = target[20].to(int)

        image2, target2 = None, 1 - new_target
        while target2 != new_target:
            idx_ = np.random.randint(len(super()))
            image2, target2 = super().__getitem__(idx_)
            target2 = target2['attributes'] == 1
            target2 = target2[20]

        return image, image2, new_target
