from torch.utils.data import Dataset
from PIL import Image

class SubsetWithTransform(Dataset):
    def __init__(self, base_ds, indices, transform):
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base_ds[self.indices[i]]
        return self.transform(img), label