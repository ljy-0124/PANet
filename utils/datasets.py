import torch
from torch.utils.data import Dataset
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import glob
import random


class TripletDataset(Dataset):
    def __init__(self, root_dir, shape=224):
        self.root_dir = root_dir
        self.shape = (shape, shape)
        num_classes = len(os.listdir(self.root_dir))
        self.train_samples = self.get_samples(list(range(num_classes)))

    # 获取指定类别的样本路径
    def get_samples(self, classes):
        # samples用于存储所有样本的路径
        samples = []
        for class_id in classes:
            # clss_id = 0,1,2,3,4,5,...,215
            class_dir = os.path.join(self.root_dir, str(class_id))
            # 遍历 0,1,2,3,...,215文件下的所有图片，并存储在class_samples列表中
            class_samples = [os.path.join(class_dir, filename) for filename in os.listdir(class_dir)]
            samples.extend(class_samples)
        return samples

    def __getitem__(self, index):
        # 文件夹index=0,0类别的路径
        anchor_path = self.train_samples[index]
        anchor_image = self.load_image(anchor_path)
        # anchor_label = 0，即标签是0代表0这个类别
        anchor_label = int(os.path.basename(os.path.dirname(anchor_path)))

        # 从0这个文件夹下取相同类别的图片作为正样本
        positive_candidates = [path for path in self.train_samples if
                               int(os.path.basename(os.path.dirname(path))) == anchor_label]
        positive_path = random.choice(positive_candidates)
        positive_image = self.load_image(positive_path)

        # 从非0文件夹下选取不同类别的图片作为负样本
        negative_candidates = [path for path in self.train_samples if
                               int(os.path.basename(os.path.dirname(path))) != anchor_label]
        negative_path = random.choice(negative_candidates)
        negative_image = self.load_image(negative_path)

        return anchor_image, positive_image, negative_image

    def __len__(self):
        return len(self.train_samples)

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(self.shape),  # Resize image to desired size
            transforms.ToTensor()  # Convert image to tensor
        ])
        image_tensor = preprocess(image)
        return image_tensor


class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
        self.image_paths = []
        self.labels = []

        for class_dir in self.class_dirs:
            class_label = int(os.path.basename(class_dir))
            class_image_paths = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
            # 获取每个类别的图像路径
            self.image_paths.extend(class_image_paths)
            # 获取每个类别的图像对应的标签
            self.labels.extend([class_label] * len(class_image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        # 返回转换后的图像对象和对应的标签
        return image_path,image, label

class LibraryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
        self.image_paths = []
        self.labels = []

        for class_dir in self.class_dirs:
            class_label = int(os.path.basename(class_dir))
            class_image_paths = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
            # 获取每个类别的图像路径
            self.image_paths.extend(class_image_paths)
            # 获取每个类别的图像对应的标签
            self.labels.extend([class_label] * len(class_image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        # 返回转换后的图像对象和对应的标签
        return image, label


# class SameDataset1(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.class_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
#
#     def __getitem__(self, index):
#         class_dir = self.class_dirs[index]
#         # 加载两张图片
#         # print("class_dir:",class_dir)
#         image_files = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
#
#         image1 = Image.open(image_files[0]).convert('RGB')
#         image2 = Image.open(image_files[1]).convert('RGB')
#         # print("iamges1:", image1)
#         # print("iamges2:", image2)
#
#         # 应用图像转换操作
#         if self.transform is not None:
#             image1 = self.transform(image1)
#             image2 = self.transform(image2)
#             # print("iamges1:", image1)
#             # print("iamges2:", image2)
#
#         return image1, image2
#
#     def __len__(self):
#         return len(self.class_dirs)


# class SameDataset(Dataset):
#     def __init__(self, data_dir, transform=None, num_samples=10):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.class_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
#         self.num_samples = num_samples
#
#     def __getitem__(self, index):
#         class_dir = self.class_dirs[index]
#         image_files = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
#
#         pairs = []
#         for _ in range(self.num_samples):
#             # 随机选择两张不同的图片
#             image1_path, image2_path = random.sample(image_files, 2)
#             if image1_path == image2_path:
#                 image2_path = random.choice(image_files)
#             # print("_________________")
#             # print("image1:", image1_path)
#             # print("image2:", image2_path)
#
#             image1 = Image.open(image1_path).convert('RGB')
#             image2 = Image.open(image2_path).convert('RGB')
#
#             if self.transform is not None:
#                 image1 = self.transform(image1)
#                 image2 = self.transform(image2)
#
#             pairs.append(image1)
#             pairs.append(image2)
#
#         return pairs
#
#     def __len__(self):
#         return len(self.class_dirs)
#
# class DifferentDataset1(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.image_files = self.get_image_files()
#
#     def get_image_files(self):
#         image_files = []
#         for root, dirs, files in os.walk(self.data_dir):
#             for file in files:
#                 if file.endswith(".jpg") or file.endswith(".png"):
#                     image_files.append(os.path.join(root, file))
#         return image_files
#
#     def __getitem__(self, index):
#         # 随机选择两个不同的图片
#         image1_path = random.choice(self.image_files)
#         image2_path = random.choice(self.image_files)
#         while image2_path == image1_path:
#             image2_path = random.choice(self.image_files)
#
#         # 加载图像
#         image1 = Image.open(image1_path).convert('RGB')
#         image2 = Image.open(image2_path).convert('RGB')
#
#         # 应用图像转换操作
#         if self.transform is not None:
#             image1 = self.transform(image1)
#             image2 = self.transform(image2)
#
#         return image1, image2
#
#     def __len__(self):
#         return len(self.image_files)
#
# class DifferentDataset(Dataset):
#     def __init__(self, data_dir, transform=None, num_samples=1000):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.image_files = self.get_image_files()
#         self.num_samples = num_samples
#
#     def get_image_files(self):
#         image_files = []
#         for root, dirs, files in os.walk(self.data_dir):
#             for file in files:
#                 if file.endswith(".jpg") or file.endswith(".png"):
#                     image_files.append(os.path.join(root, file))
#         return image_files
#
#     def __getitem__(self, index):
#         # 随机选择两个不同的图片
#         image1_path = random.choice(self.image_files)
#         image2_path = random.choice(self.image_files)
#         while image2_path == image1_path:
#             image2_path = random.choice(self.image_files)
#
#
#         # 加载图像
#         image1 = Image.open(image1_path).convert('RGB')
#         image2 = Image.open(image2_path).convert('RGB')
#
#         # 应用图像转换操作
#         if self.transform is not None:
#             image1 = self.transform(image1)
#             image2 = self.transform(image2)
#
#         return image1, image2
#
#     def __len__(self):
#         return self.num_samples

if __name__ == '__main__':
    root_dir = "/root/program/metric_demo/data/train"

    train_dataset = TripletDataset(root_dir)

    # Get a random triplet
    # 从train_dataset中随机选择一个三元组anchor_image, positive_image, negative_image
    anchor_image, positive_image, negative_image = train_dataset[random.randint(0, len(train_dataset))]

    # Convert tensors to PIL images
    anchor_image_pil = TF.to_pil_image(anchor_image)
    positive_image_pil = TF.to_pil_image(positive_image)
    negative_image_pil = TF.to_pil_image(negative_image)

    # Plot the images
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(anchor_image_pil)
    ax[0].set_title("Anchor")
    ax[0].axis('off')

    ax[1].imshow(positive_image_pil)
    ax[1].set_title("Positive")
    ax[1].axis('off')

    ax[2].imshow(negative_image_pil)
    ax[2].set_title("Negative")
    ax[2].axis('off')

    plt.show()
