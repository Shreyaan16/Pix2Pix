import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path)
        
        # Split the image into input and target
        w, h = image.size
        input_image = image.crop((0, 0, w//2, h))  # Left half
        target_image = image.crop((w//2, 0, w, h))  # Right half

        # Apply resize to both
        input_image = config.both_transform(input_image)
        target_image = config.both_transform(target_image)

        # Apply random seed for synchronized augmentation (horizontal flip)
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        
        # Check if we should flip (to keep both synchronized)
        should_flip = torch.rand(1).item() < 0.5
        
        if should_flip:
            from torchvision.transforms import functional as TF
            input_image = TF.hflip(input_image)
            target_image = TF.hflip(target_image)
        
        # Apply color jitter only to input
        input_image = config.transform_only_input(input_image)
        
        # Apply normalization to target (no augmentation)
        target_image = config.transform_only_mask(target_image)

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("data/maps/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys
        sys.exit()