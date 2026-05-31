import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def show5(img_loader):
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    
    dataiter = iter(img_loader)

    # batch[0] (images).shape → (5, C, H, W) 画像：5枚まとめて
    # batch[1] (labels).shape → (5,) ラベル：5個まとめて
    batch = next(dataiter)

    # 中身はクラス番号: tensor([3, 0, 8, 1, 6])
    labels = batch[1][0:5]

    # 最初の5枚だけ取り出している: (5, 3, 32, 32)
    images = batch[0][0:5]
    
    for i in range(5):
        print(classes[labels[i]])
        image = images[i].numpy()
        # 次元（axis）の順番を (1番目, 2番目, 0番目) に入れ替える
        # imshowで扱うため(高さ, 幅, チャンネル) = (H, W, C)にしている
        # (チャンネル, 高さ, 幅) (0, 1, 2) -> (高さ, 幅, チャンネル) (1, 2, 0)
        plt.imshow(image.transpose(1, 2, 0))
        plt.show()

# Create the training dataset
training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())

# Create the training dataloader with batch size 5
train_loader = DataLoader(training_data, batch_size=5)

# View 5 images using the show5 function
show5(train_loader)