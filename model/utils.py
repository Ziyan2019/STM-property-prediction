import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random

# Global constants
HEIGHT, WIDTH = 0, 0  # These appear unused - consider removing if not needed


def read_xyz(filename):
    """Read XYZ file and extract atom information"""
    with open(filename) as f:
        raw = f.readlines()

    result = []
    for line in raw:
        line = line.strip()
        if not line or line[0] not in ['C', 'H']:
            continue
        parts = line.split()
        atom = parts[0]
        locs = list(map(float, parts[1:4]))  # Only take first 3 coordinates if more exist
        result.append([atom] + locs)
    return result


def process_raw_image(filename, resize_size, raw_size):
    """Process raw image into grayscale and edge detection channels"""
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {filename}")

    img = cv2.resize(img, raw_size)
    _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, threshold1=100, threshold2=200)

    # Calculate padding for centering
    original_h, original_w = img.shape
    new_w, new_h = resize_size
    top = (new_h - original_h) // 2
    bottom = new_h - original_h - top
    left = (new_w - original_w) // 2
    right = new_w - original_w - left

    # Apply padding
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=0)
    edges_padded = cv2.copyMakeBorder(edges, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=0)

    return np.stack([img_padded, edges_padded])


def gaussian_kernel(size, sigma=None):
    """Generate 2D Gaussian kernel"""
    sigma = sigma or 0.3 * ((size - 1) * 0.5 - 1) + 0.8  # Default sigma calculation
    kernel_1d = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d.T)
    return (kernel_2d - np.min(kernel_2d)) / np.max(kernel_2d) * 255


def process_locs(locs, resize_size, kernel_size=15):
    """Convert atom locations to a 2D density map"""
    target_image = np.zeros(resize_size[::-1])  # Create with (height, width) shape

    for loc in locs:
        if loc[0] == 'H':  # Skip hydrogen atoms
            continue

        # Scale and shift coordinates
        x, y = loc[1] + resize_size[0] / 20, loc[2] + resize_size[1] / 20
        x, y = int(10 * x), int(10 * y)

        # Calculate kernel bounds with boundary checks
        half_size = kernel_size // 2
        x_start = max(0, x - half_size)
        x_end = min(resize_size[0], x + half_size + 1)
        y_start = max(0, resize_size[1] - 1 - (y + half_size))
        y_end = min(resize_size[1], resize_size[1] - (y - half_size))

        # Apply Gaussian kernel
        kernel = gaussian_kernel(kernel_size)
        target_image[y_start:y_end, x_start:x_end] = np.maximum(
            kernel, target_image[y_start:y_end, x_start:x_end]
        )

    return target_image


class I2IDataset(Dataset):
    """Dataset class for image-to-image translation"""

    def __init__(self, folders):
        self.labels = {}
        self.samples = []

        for folder in folders:
            if not os.path.isdir(folder):
                continue

            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if not os.path.isdir(item_path):
                    continue

                img_files = []
                xyz_file = None

                for file in os.listdir(item_path):
                    file_path = os.path.join(item_path, file)
                    if file.endswith('.xyz'):
                        xyz_file = file_path
                    elif file[-11].isdigit() and int(file[-11]) >= 7:  # Check file naming pattern
                        img_files.append(file_path)

                if xyz_file and img_files:
                    self.labels[item_path] = read_xyz(xyz_file)
                    self.samples.append((item_path, sorted(img_files)))

    def __getitem__(self, index):
        item_id, img_files = self.samples[index]
        resized_size = (640, 640)
        raw_size = (300, 200)

        # Process all images for this sample
        images = [process_raw_image(f, resized_size, raw_size) for f in img_files]
        data = np.vstack(images) / 255.0

        # Process label
        label = process_locs(self.labels[item_id], resized_size) / 255.0
        label = label[np.newaxis, ...]  # Add channel dimension

        return (torch.from_numpy(data.astype(np.float32)),
                torch.from_numpy(label.astype(np.float32)))

    def __len__(self):
        return len(self.samples)


class VecDataset(Dataset):
    """Dataset class for pre-processed vector data"""

    def __init__(self, folder):
        if not os.path.isdir(folder):
            raise ValueError(f"Directory {folder} does not exist")

        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith('.pt')
        ]

    def __getitem__(self, index):
        data = torch.load(self.files[index])
        num_pairs = (len(data) - 1) // 2
        idx = random.randint(0, num_pairs - 1)
        return (
            data[idx * 2: idx * 2 + 2],  # Input pair
            data[-1].unsqueeze(0)  # Label
        )

    def __len__(self):
        return len(self.files)


def save_dataset_as_vectors(dataset, target_dir='./datasets/vector'):
    """Save dataset as individual vector files"""
    os.makedirs(target_dir, exist_ok=True)

    for i, (data, label) in enumerate(tqdm(dataset, desc="Saving dataset")):
        combined = torch.cat([data, label], dim=0)
        torch.save(combined, os.path.join(target_dir, f'item_{i:04d}.pt'))

    print(f'Dataset saved to {target_dir}')


if __name__ == '__main__':
    # Example usage
    dataset = I2IDataset(['../data/stm_image_1148/', '../data/stm_image_908/'])
    save_dataset_as_vectors(dataset)

    # Test loading
    vec_dataset = VecDataset('./datasets/vector')
    data, label = vec_dataset[0]

    print(f"Data sum: {torch.sum(data[0])}, Label sum: {torch.sum(label)}")
    print(f"Shapes - Data: {data.shape}, Label: {label.shape}")

    # Save sample images for inspection
    cv2.imwrite('./label.png', (label[0].numpy() * 255).astype(np.uint8))
    cv2.imwrite('./img.png', (data[1].numpy() * 255).astype(np.uint8))
