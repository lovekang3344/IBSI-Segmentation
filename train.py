from glob import glob
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.utils import make_grid, save_image

from 分组作业6.model import UNet
import os
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from 分组作业6.mydataset import myDataset
from 分组作业6.utils import matrix


def Train_Unet(device, data_path, lr, model_path, batch_size, epochs, what='ConvTranspose2d'):
    os.makedirs(f'./checkpoint/{what}_result', exist_ok=True)
    unet = UNet(1, 2, False if what == 'ConvTranspose2d' else True)
    # unet.load_state_dict(torch.load(model_path))
    unet.to(device)
    train_dataset = myDataset(data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter('./logs')
    opt = optim.Adam(unet.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    iter = 0
    for epoch in range(epochs):
        unet.train()
        for image, label in train_loader:
            opt.zero_grad()
            image, label = image.to(device, dtype=torch.float32).unsqueeze(1), label.to(device, dtype=torch.int64)
            pred = unet(image)
            loss = criterion(pred, label)
            output = F.sigmoid(pred).max(dim=1)[1]
            _, output = torch.max(pred, dim=1)
            output = output.cpu().detach()
            mat = matrix(output, label)
            loss.backward()
            opt.step()
            iter += 1
            print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - mIoU: {mat.mIoU:.4f} - mPA: {mat.mPA:.4f}')
            writer.add_scalar(f'Loss/{what}', loss.item(), iter)
            writer.add_scalar(f'mIoU/{what}', mat.mIoU, iter)
            writer.add_scalar(f'mPA/{what}', mat.mPA, iter)
        torch.save(unet.state_dict(), model_path)

        unet.eval()
        with torch.no_grad():
            pred_imgs = []
            for img_path in glob(data_path.replace('train', 'val') + '*.png'):
                # 读入照片，(h, w) --> (1, 1, h, w)
                img = torch.tensor(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                pred = unet(img)
                output = F.sigmoid(pred).max(dim=1)[1]
                output = output.cpu().detach().to(torch.uint8)
                output[output == 1] = 255
                # output = output.unsqueeze(0)
                # plt.figure()
                # plt.imshow(output.squeeze(), cmap='gray')
                # plt.axis('off')
                # plt.show()
                result = np.transpose(output.numpy(), (1, 2, 0)).astype(np.uint8)
                cv2.imwrite(f'./checkpoint/{what}_result/{epoch}_'+img_path.split('\\')[-1], result)
                pred_imgs.append(output)
            writer.add_image(f'val/{what}', make_grid(pred_imgs, padding=4), iter)



if __name__ == '__main__':
    data_path = './data/train/image/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('./checkpoint/UNet/', exist_ok=True)
    lr = 1e-4
    epochs = 6
    batch_size = 4
    model_path = './checkpoint/UNet/ConvTranspose2d_model.pth'
    Train_Unet(device, data_path, lr, model_path, batch_size, epochs)
    model_path = './checkpoint/UNet/Upsampling_model.pth'
    Train_Unet(device, data_path, lr, model_path, batch_size, epochs, what='Upsampling')
