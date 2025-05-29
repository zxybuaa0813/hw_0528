# 基于CNN实现mnist手写数据集识别

import torch
from jinja2.optimizer import optimize
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


batch_size = 64
# 将图片格式转为Tensor格式 (28,28) --> (1,28,28)
# 使用数据集的均值0.1307和标准差0.3081对数据进行标准化
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
# 读取数据集
train_dataset = datasets.MNIST(root = "../dataset/mnist",train = True,download = True,transform = transform)
train_iter = DataLoader(train_dataset,shuffle = True,batch_size=batch_size)
test_dataset = datasets.MNIST(root = "../dataset/mnist",train = False,download = True,transform = transform)
test_iter = DataLoader(test_dataset,shuffle = True,batch_size=batch_size)

# 定义网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.ll = torch.nn.Linear(320,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pooling(x)
        x = F.relu(self.conv2(x))
        x = self.pooling(x)
        x = x.view(x.size(0),-1)
        x = self.ll(x)
        return x

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1:  # 判断是否为多类别预测
        y_hat = y_hat.argmax(dim=1)
    correct = (y_hat == y).float()
    total_correct = correct.sum().item()  # 求和并转为Python数值

    return total_correct

def evaluate_accuracy_gpu(net,data_iter,device = None):
    if isinstance(net,nn.Module):
        net.eval()

    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X),y),y.numel())
        return metric[0]/metric[1]

# 训练过程
def train(net,train_iter,test_iter,lr,device,epochs = 400):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training_on',device)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_losses = []
    train_accs = []
    test_accs = []

    for ep in range(epochs):
        metric = Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter,0):
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0],accuracy(y_hat,y),X.shape[0])
            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]

        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        train_losses.append(train_l)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'Epoch {ep + 1}/{epochs}, Train Loss: {train_l:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    return train_losses, train_accs, test_accs


def plot_training_metrics(train_losses, train_accs, test_accs):
    epochs = len(train_losses)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, 'b-', label='Train Acc')
    plt.plot(range(1, epochs + 1), test_accs, 'r-', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_predictions(net, test_dataset, num_images=10, device=device):
    net.eval()
    fig, axes = plt.subplots(nrows=2, ncols=num_images // 2, figsize=(15, 4))

    with torch.no_grad():
        # 随机选取num_images张测试图像
        indices = torch.randint(0, len(test_dataset), (num_images,))
        for i, idx in enumerate(indices):
            img, true_label = test_dataset[idx]
            img = img.unsqueeze(0).to(device)  # 添加批次维度
            output = net(img)
            pred_label = output.argmax(dim=1).item()

            # 显示图像
            ax = axes[i // 5, i % 5]  # 2行5列布局
            ax.imshow(img.squeeze().cpu(), cmap='gray')
            ax.set_title(f"true:{true_label}\npred:{pred_label}", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 训练并绘图
    train_losses, train_accs, test_accs = train(
        net = model,
        train_iter = train_iter,
        test_iter = test_iter,
        lr = 0.01,
        device = device,
        epochs = 10
    )

    plot_training_metrics(train_losses, train_accs, test_accs)

    visualize_predictions(net = model, test_dataset = test_dataset, num_images=10, device=device)

















