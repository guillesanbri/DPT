import os
import cv2
import numpy as np
import util.io
from torch.utils.data import Dataset


class KITTIDataset(Dataset):
    def __init__(self, img_dir, depth_dir, filenames_file, transform):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        with open(filenames_file) as f:
            image_paths = []
            depth_paths = []
            for file in f.readlines():
                file = file.strip()
                parts = file.split(" ")
                camera = "image_02" if parts[2] == "l" else "image_03"
                image_path = f"{parts[0]}/{camera}/data/{parts[1].zfill(10)}.png"
                depth_path = f"{parts[0].split('/')[1]}/proj_depth/groundtruth/{camera}/{parts[1].zfill(10)}.png"
                image_paths.append(image_path)
                depth_paths.append(depth_path)
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load raw image
        img_path = os.path.join(self.img_dir, self.image_paths[idx])
        image = util.io.read_image(img_path)
        # Load depth image
        depth_subdir = os.path.join(self.depth_dir, "train")
        depth_path = os.path.join(depth_subdir, self.depth_paths[idx])
        try:
            depth = cv2.imread(depth_path, -1)
            if depth is None:
                raise AttributeError("File not found in train folder, looking in val folder")
        except AttributeError:
            depth_subdir = os.path.join(self.depth_dir, "val")
            depth_path = os.path.join(depth_subdir, self.depth_paths[idx])
            depth = cv2.imread(depth_path, -1)
        depth = depth / (256 ** 2)
        depth = depth[:, :, np.newaxis].repeat(3, axis=2)
        # Apply transformations as needed
        transformed_images = self.transform({"image": image, "depth": depth})
        image = transformed_images["image"]
        depth = transformed_images["depth"]
        return image, depth
