from torch.utils.data import Dataset

from PIL import Image

class WithTransforms(Dataset):
    def __init__(self, images: list[Image.Image], transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        return self.transform(image)

class SubsetWithTransform(Dataset):
    def __init__(self, base_ds, indices, transform):
        self.base_dataset = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, label = self.base_dataset[self.indices[index]]
        return self.transform(img), label