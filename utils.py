import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
import seaborn as sns
from matplotlib import pyplot as plt

# x = torch.tensor([[0, 1, 2], [0, 0, 0]])
# label = torch.tensor([[0, 1, 1], [2, 0, 0]])
# x = torch.randint(0, 2, (2, 28, 28))
# label = torch.randint(0, 2, (2, 28, 28))

class matrix:
    def __init__(self, pred, label):
        self.pred = pred.numpy()
        self.label = label.cpu().detach().numpy()
        batch_size = self.pred.shape[0]
        self.mIoU = 0
        self.mPA = 0

        for i in range(batch_size):
            cm = self.cal_cm(self.pred[i], self.label[i])
            intersection = np.diag(cm)
            sum_c = np.sum(cm, axis=0)
            sum_r = np.sum(cm, axis=1)
            Union = sum_c + sum_r - intersection
            IoU = intersection / Union
            CPA = np.diag(cm) / sum_c
            # 忽略背景
            self.mIoU += sum(IoU[1:])
            self.mPA += sum(CPA[1:])

        self.mIoU /= batch_size
        self.mPA /= batch_size

    def cal_cm(self, pred, label):
        pred = pred.reshape(1, -1).squeeze()
        label = label.reshape(1, -1).squeeze()
        cm = metrics.confusion_matrix(label, pred)
        return cm