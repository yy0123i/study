##### 网络结构

![image-20240223144200899](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223144200899.png)

![image-20240223144209248](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223144209248.png)

![image-20240223153253029](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223153253029.png)

第一个卷积，生成32个通道，每个通道是个 5 * 5 * 3的filter，生成梯度准备进行所有参数的更新。

第一个参数是输入的通道数，他决定了我们的卷积核第三维度。所以filter维度为

3，那么filter为5 * 5 * 3，那么32 * 32 * 3与每个filter生成一个特征图（一个特征图的维度为32 * 32，并没有深度，三个维度的卷积核结果相加，呈现在一个格子中），第二个参数32,代表生成32个特征图。

padding = 2, 那么 32 - 5 - 2 * 2 + 1 = 32 , 所以生成后的为 32 * 32 * 32

其他并不难。

##### 交叉熵损失函数

真实的交叉熵

样本实际的分类为c
$$
L = -(check(i)* log(p_i))
一个样例为(0.1,0.8,0.4)
如果这个样例实际为0,
那么有 log0.1 + log0.2 + log0.6  
解释\\
如果本来预测的很好,那么本来那项趋于0. 如果其他项预测值很大,那么 log*(1-p),即他的概率越大，反而造成的损失越多。这是符合实际的。
$$
pytorch对这一函数进行改进，为了拟合预测错误的程度。

![image-20240223145653426](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223145653426.png)



分类正确，前面的

-x[class] 会比较小，趋于-1，那么后面的项，如果预测成功，而别的值也很大后面就会很大，这就说明区分性不强，所以增加了loss。

如果后面的值只有预测正确的值大，其他很小，说明区分性很强，那么其值较小，总体loss较低。

![image-20240223145901975](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223145901975.png)

C是类别。
$$
-x + ln(x + u)   
$$


##### 优化器

随机梯度下降就是一种优化器，通过优化器对参数更新来趋于最小值。

```python
for input, target in dataset:
    optimizer.zero_grad() 梯度清0
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step() 根据梯度，利用优化器完成模型参数的更新.
```

##### epoch

对于同一个数据学习次数。

##### 现有网络改写

```
vgg16 = torchvision.models.vgg16(pretrained=False)  # 分成1000类
vgg16_true = torchvision.models.vgg16(pretrained=True) # 在网络上下载训练的参数
dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)   # 分为10类.
# 1.可以再加一层线性层
# 2. vgg_16.classifier(看具体在哪个部分加).add_module('名字', nn.Linear(1000,10))
```

##### 网络模型的保存与读取

保存模型参数和结构

![image-20240223165326044](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223165326044.png)

读取

![image-20240223165502088](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223165502088.png)

保存模型参数(官方推荐)

![image-20240223165735789](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223165735789.png)

读取应当先建立对应模型，因为没有保存模型结构，然后再把参数传递给模型

![image-20240223170046913](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223170046913.png)

##### 展示图像

创建

![image-20240223171614136](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223171614136.png)

tensorboard --logdir=logs_train

writer.add_scalar("标题", x , y)

画图.

##### GPU训练

![image-20240223172631441](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223172631441.png)

优化器没有cuda

![image-20240223172648166](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223172648166.png)

![image-20240223172709027](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223172709027.png)

第二种方法

![image-20240223172854337](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223172854337.png)

![image-20240223172908330](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223172908330.png)

![image-20240223172928946](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223172928946.png)

##### 模型检验

[PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=05469f7acaa60fdddb57183e7dae7f06)

##### 训练集，测试集，验证集

训练集是训练数据过程中，不断调整权重参数，得到模型，参与梯度下降。

验证集是修改超参数，让训练集训练后在验证集能够表现好，验证集不参与训练，调节了泛化性能。为了即时检测模型训练，可以提前结束。防止过拟合。

测试集，因为验证集参与了调整超参数，那么这个超参数一定是适合验证集的，但是是否适应其他集合数据？所以我们要用一个从未使用的数据集，测试集来作最后的验证

![image-20240223181445849](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223181445849.png)
