import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim
import func


# 残差连接模块
class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 层归一化模块
class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self,x,**kwargs):
        return self.fn(self.norm(x), **kwargs)

# 前馈网络模块
class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

# multi head 模块
class Attention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim,dim)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads   # 解包 shape:batch_size,sequence_length,feature_dim, 注意力头的数量
        qkv = self.to_qkv(x)   # 将x映射到q,k,v三个矩阵，形状是[b,n,dim*3]
        # 先把最后一个维度变成3个维度(3,h,dim//h)  再重新排列
        # q,k,v 的形状均为[b,h,n,d]
        q, k, v = rearrange(qkv,'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)
        dots = torch.einsum('bhid,bhjd->bhij',q,k) * self.scale  # dots的维度变[b,h,n,n]

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1,0) ,value = True)   # mask第一维展开，在最后一维上 左侧填充一列True
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimension'
            mask = mask[:,None,:] * mask[:,:,None]
            dots.masked_fill_(~mask, float('-inf'))   # ~mask中True会被填充为‘-inf’,即mask中False会被填充为'-inf',经过softmax之后就会变为0
            del mask

        attn = dots.softmax(dim = -1)

        out = torch.einsum('bhij,bhjd->bhid',attn,v)   # [b h n n]*[b h n d] -> [b h n d]
        out = rearrange(out,'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

# transformer 模块
class Transformer(nn.Module):
    # dim:输入特征维度  depth:Transformer块的数量  mlp_dim:ffn的隐藏层维度
    def __init__(self,dim,depth,heads,mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Residual(PreNorm(dim,Attention(dim,heads = heads))),
                    Residual(PreNorm(dim,FeedForward(dim,mlp_dim)))
                ])
            )

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x,mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self,*,image_size,patch_size,num_classes,dim,depth,heads,mlp_dim,channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, '报错：图像没有被patch_size完美分割'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2    # 每个patch被展平后的实际长度

        self.patch_size = patch_size
        self.pos_embedding =  nn.Parameter(torch.randn(1,num_patches+1,dim))  # +1 是因为有cls_token
        self.patch_to_embedding = nn.Linear(patch_dim,dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim,num_classes)
        )

    def forward(self,img,mask = None):
        # print('init',img.shape)
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1 = p, p2 =p) # 将二维图像分割为多个patch，(h w)合并为patch数量，(p1 p2 c)每个patch展平为一维向量
        # print('rearrange:',x.shape)
        x = self.patch_to_embedding(x)
        # print('patch_embedding:',x.shape)
        cls_tokens = self.cls_token.expand(img.shape[0],-1,-1)  # 将第0维扩展到img.shape[0]维，其余两个维度保持原样
        # print('cls_tokens:',cls_tokens.shape)
        x = torch.cat((cls_tokens,x),dim = 1)
        # print('cat cls:',x.shape)
        x += self.pos_embedding
        x = self.transformer(x, mask)
        # print('after transformer', x.shape)
        x = self.to_cls_token(x[:,0])   # 取出所有批次样本的cls_token
        # print('cls_token',x.shape)
        y = self.mlp_head(x)
        # print('mlp_head',y.shape)
        return y

def train_epoch(model,optimizer,data_loader,loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i,(data,target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data),dim=1)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()

    if i % 100 == 0:
        print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                   ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                   '{:6.4f}'.format(loss.item()))
        loss_history.append(loss.item())

def train(model,optimizer,train_iter,test_iter,device,num_epochs=20):
    model.to(device)
    train_losses = []
    train_accs = []
    test_accs = []

    for ep in range(num_epochs):
        metric = func.Accumulator(3)
        model.train()

        for i, (data, target) in enumerate(train_iter):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # 这里的output可能不对
            with torch.no_grad():
                metric.add(loss * data.shape[0], func.accuracy(output,target),data.shape[0])

            train_l = metric[0]/metric[2]
            train_acc = metric[1]/metric[2]

        test_acc = func.evaluate_accuracy_gpu(model, test_iter, device)
        train_losses.append(train_l)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'Epoch {ep + 1}/{num_epochs}, Train Loss: {train_l:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    return train_losses, train_accs, test_accs




batch_size = 64
# 将图片格式转为Tensor格式 (28,28) --> (1,28,28)
# 使用数据集的均值0.1307和标准差0.3081对数据进行标准化
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
# 读取数据集
train_dataset = datasets.MNIST(root = "../dataset/mnist",train = True,download = True,transform = transform)
train_iter = DataLoader(train_dataset,shuffle = True,batch_size=batch_size)
test_dataset = datasets.MNIST(root = "../dataset/mnist",train = False,download = True,transform = transform)
test_iter = DataLoader(test_dataset,shuffle = True,batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1, dim=64, depth=6, heads=8, mlp_dim=128)
optimizer = optim.Adam(model.parameters(),lr = 0.005)
train_loss_history,test_loss_history = [],[]


if __name__ == '__main__':
    # 训练并绘图
    train_losses, train_accs, test_accs = train(
        model = model,
        optimizer = optimizer,
        train_iter = train_iter,
        test_iter = test_iter,
        device = device,
        num_epochs = 10
    )

    func.plot_training_metrics(train_losses, train_accs, test_accs)

    func.visualize_predictions(net = model, test_dataset = test_dataset, num_images=10, device=device)










