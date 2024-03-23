##### 数字识别入门

举个例子：将10kg的面粉使用面条加工机(每次只能处理2kg)，加工成10kg的面条。首先得把10kg面粉分成5份2kg的面粉，然后放入机器加工，经过5次，可以将这10kg面粉首次加工成面条，但是现在的面条肯定不好吃，因为不劲道，于是把10kg面条又放进机器再加工一遍，还是每次只处理2kg，处理5次，现在感觉还行，但是不够完美；于是又重复了一遍：将10kg上次加工好的面条又放进机器，每次2kg，加工5次，最终成型了，完美了，结束了。那么到底重复加工几次呢？只有有经验的师傅才知道。

这就形象地说明：Epoch就是10斤面粉被加工的次数（上面的3次）Batch Size就是每份的数量（上面的2kg），Iteration就是将10kg面粉加工完一次所使用的循环次数（上面的5次）。显然 1个epoch  = BatchSize * Iteration
链接：https://www.zhihu.com/question/27700702/answer/941067214

epoch

总共被加工的次数，所有数据

iteration

总共的数据被训练一次需要循环的次数， 总共的数据训练一次，更新了iteration次权重 。

batch_size

每次处理数据的大小

相当于有5000个图片，那么batch = 100，每次处理50个，也就是我们训练模型从来都没有直接计算5000个图片的损失函数，而是每次放入50个图片，然后计算损失函数，更新权重。

batch_size = 64

![image-20240224132057621](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240224132057621.png)

可以看到我们拿了64张图片去计算损失函数，然后梯度下降更新权重，而不是一张图片去更新权重。

如果batch = 1，那么我们每次根据某张图片的梯度方向进行更新，可能并不符合全局，很难到达收敛的效果。

如果batch=n，这些图片求出的loss是n张图片的平均loss，他的梯度更能够反应总体的梯度方向，更容易达到收敛。

batch = 全集，每次用所有图片进行训练，导致所有图片被加载到内存运算，对于计算机的压力很大，并且我们只要选取总体的一部分就可以找到合适的梯度方向，总体带来的开销比较大。

##### 关于python引入

我们引入number中的Net，但是发现他执行了number的代码

![image-20240224140959010](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240224140959010.png)

由于python是脚本语言，import number会扫描number的所有语句，当遇到语句时会发生执行。所以我们如果把语句写入

if __name__ == __main__  那么执行语句，就不会执行。

当我们执行入口是当前文件时__name__ = main,其他情况下为文件名。

所以我们执行test，只有在test中__name__ = main,当进入number扫描时，__name__  = number, 不等于__main__ 所以不会执行语句。

##### 载入一个图片，并预测

```python
img = Image.open('data/1.jpg')
transform = transforms.Grayscale() # 灰度化
img = transform(img)
# transform = transforms.RandomRotation(degrees=(90, 90))
transform = transforms.Resize((28, 28)) # 更改大小
img = transform(img)
plt.imshow(img, cmap=plt.get_cmap('gray')) # 绘制矩阵
transform = transforms.ToTensor()   # 转化为张量形式
img = transform(img)
model = torch.load('./model.pth')
net = Net()
net.load_state_dict(model)       
img = img.view(1, 1, 28, 28)     # 切换输入方式 (size, c, h , w)
output = net(img)                # [p1,p2 ..... ]
print(torch.max(output, 1))      # [取最大值的索引.]
```

##### 全部代码展示

```python
import torch
import torchvision
from torch import nn

from torch.utils.data import DataLoader
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch = 10
batch_size_train = 128
batch_size_test = 1000
lr = 0.01
monentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

# 加载训练集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/see', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

# 加载测试集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(30, 60, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2940, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 2940)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


print(__name__)
if __name__ == "__main__":
    network = Net()
    network = network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=monentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(epoch + 1)]
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)


    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            # 每10次batch, 记录一个模型.
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )
                torch.save(network.state_dict(), './model.pth')
                torch.save(optimizer.state_dict(), './optimizer.pth')


    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                output = network(data)
                test_loss += loss_fn(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset))
        )


    for epoch in range(1, epoch + 1):
        train(epoch)
        test()

```

