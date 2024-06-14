import random
import torch
import torch.nn as nn
from glob import glob
import cv2
from torch.utils.data import Dataset, DataLoader


class myDataset(Dataset):
    def __init__(self, data_path):
        super(myDataset, self).__init__()
        self.data_path = data_path
        self.img_path = glob(data_path + '*.png')

    def augument(self, image, flipcode):
        flip = cv2.flip(image, flipcode)
        return flip

    def __getitem__(self, index):
        img_path = self.img_path[index]
        label_path = img_path.replace('image', 'label')
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if label.max() > 1:
            label = label / 255
        # 图像增强, 垂直翻转0，水平翻转1，水平和垂直翻转-1
        flipcode = random.choice([-1, 0, 1, 2])
        if flipcode != 2:
            image = self.augument(image, flipcode)
            label = self.augument(label, flipcode)
        return image, label

    def __len__(self):
        return len(self.img_path)

# if __name__ == '__main__':
#     data_path = r"./data/train/image/"
#     dataset = myDataset(data_path)
#     # print(dataset)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
#     print(next(iter(dataloader)))

