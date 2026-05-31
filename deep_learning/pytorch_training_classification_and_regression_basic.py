import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

########################################################################################
# 分類モデルのトレーニング - classification model
########################################################################################

# 分類モデル - classification model
class CIFAR_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = F.relu
        self.output = F.log_softmax
        self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output(self.fc3(x))
        return x

# クラス分類用のトレーニング
def train_network_classification(net, train_loader, test_loader):
    num_epochs = 10
    
    # Negative Log Likelihood Lossはクラス分類のための損失関数
    # 正解クラスに高い確率を出すように」モデルを訓練するためのもの
    criterion = nn.NLLLoss()

    # SGD = Stochastic Gradient Descentは最も基本的な最適化アルゴリズム
    # 重み = 重み − 学習率 × 勾配
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    train_loss_history = list()
    val_loss_history = list()

    for epoch in range(num_epochs):

        # モデルを学習モードにスイッチ
        net.train()

        train_loss = 0.0
        train_correct = 0

        for i, data in enumerate(train_loader):
            # data = [inputs, labels]
            inputs, labels = data

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # 前回計算した勾配（gradient）をリセットする処理 (PyTorchは勾配が累積するので毎バッチで消すのが基本）
            optimizer.zero_grad()

            # 入力データをモデルに通して予測を作っている、forwardメソッドが実行される
            outputs = net(inputs)

            # 損失の計算
            loss = criterion(outputs, labels)
            
            # backpropagation: 逆伝播で勾配を計算
            loss.backward()

            # 重みの更新
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)
            train_correct += (preds == labels).sum().item()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1} training accuracy: {train_correct / len(train_loader):.2f}% training_loss: {train_loss / len(train_loader):.5f}")
        train_loss_history.append(train_loss)

        val_loss = 0.0
        val_correct = 0

        # モデルを「評価モード（推論モード）」に切り替える
        net.eval()

        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, 1)
            val_correct += (preds == labels).sum().item()
            val_loss += loss.item()
        print(f'Epoch {epoch + 1} validation accuracy: {val_correct/len(test_loader):.2f}% validation loss: {val_loss/len(test_loader):.5f}')
        val_loss_history.append(val_loss)  

    # plt.plot(train_loss_history, label="Training Loss")
    # plt.plot(val_loss_history, label="Validation Loss")
    # plt.legend()
    # plt.show()

# torchvisionで使う前処理（transform）を定義
# 画像をニューラルネットワークに入力できる形に変換
# 複数の変換処理を上から順に適用するパイプラインを作る
transform = transforms.Compose([
    transforms.ToTensor(), # Tensor化（C × H × W）
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 値を [-1, 1] に正規化
])


# CIFAR10: 小さなカラー画像（32×32ピクセル）が10種類のカテゴリに分かれているデータセット
training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# classificationモデルの作成
mlp = CIFAR_MLP()
if torch.cuda.is_available():
    mlp.cuda()

# classification modelのトレーニング
train_network_classification(mlp, train_loader, test_loader)

########################################################################################
# 回帰モデルのトレーニング - regression model 数値予測のためのモデル
########################################################################################

class Housing_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = F.relu
        self.hidden = nn.Linear(8, 2)
        self.prediction = nn.Linear(2, 1)

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.prediction(x)
        return x

def train_network_regression(net, train_loader, test_loader):
    num_epochs = 10

    criterion = nn.L1Loss(reduction="sum")

    optimizer = optim.SGD(net.parameters(), lr=0.05)

    train_loss_history = list()
    val_loss_history = list()

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data

            if torch.cuda.is_available():
                inputs, lables = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f'Epoch {epoch + 1} training loss: {train_loss/len(train_loader):.5f}')
        train_loss_history.append(train_loss)

        val_loss = 0.0
        net.eval()
        for inputs, labels in test_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
        
        print(f'Epoch {epoch + 1} validation loss: {val_loss/len(test_loader):.5f}')
        val_loss_history.append(val_loss)

    # plt.plot(train_loss_history, label="Training Loss")
    # plt.plot(val_loss_history, label="Validation Loss")
    # plt.legend()
    # plt.show()

# 回帰モデルを学習させるためのデータ準備一式
data, target = fetch_california_housing(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3)

train_x = torch.tensor(train_x, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

train_california = torch.utils.data.TensorDataset(train_x, train_y)
test_california = torch.utils.data.TensorDataset(test_x, test_y)

train_loader = DataLoader(train_california, batch_size=10, shuffle=True)
test_loader = DataLoader(test_california, batch_size=10)

mlp = Housing_MLP()
if torch.cuda.is_available():
    mlp.cuda()

train_network_regression(mlp, train_loader, test_loader)
