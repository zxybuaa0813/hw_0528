import torch
from torch import nn
import matplotlib.pyplot as plt

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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



