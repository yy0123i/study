##### 神经元

![image-20240208123230905](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208123230905.png)

##### 神经网络

![image-20240208123935556](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208123935556.png)

输入层的每个指标都输入各个神经元，然后会忽略那些不需要的指标，其参数极小趋于0.

从隐藏层到输出层，相当于一开始的指标作完特征工程，然后作逻辑回归预测。所以自动完成了特征工程，选取了合适的指标来预测。但

是第二步的隐层怎么实现?

![image-20240208125307842](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208125307842.png)

隐藏层不需要人为告知，就自己知道选取什么指标。

![image-20240208125503687](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208125503687.png)

##### 数学原理细节

![image-20240208130921041](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208130921041.png)

> [1] 代表第一层的参数， [0]代表输入层。

![image-20240208135736090](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208135736090.png)

计算过程，利用上一层的a，作为下一层的指标输入。

![image-20240208140038006](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208140038006.png)

g，又叫作激活函数，因为他产生了激活值。

##### 前向传播

![image-20240208141535883](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208141535883.png)

从前向后不断传播激活向量a。

在tensorflow中，不用np的一个方括号表示，用两个方括号表示。

![image-20240208145700432](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208145700432.png)

原因是用矩阵表示一维度的时候处理效率更高。

在numpy中，用[ ], 表示向量。但是在tensorflow中，用[[]] 来表示所有，包括向量，这样作的目的是加快效率。



##### 如何构建神经网络

![image-20240208150113715](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208150113715.png)

> sequnential 完成前一个输出作为后一个输入.

输入数据形式

![image-20240208150223874](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208150223874.png)

编码形式

![image-20240208150532125](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208150532125.png)

![image-20240208161615137](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208161615137.png)

##### numpy的shape[0]

当只有一行元素时 

```python
b = np.array([1,2,3,4])
```

(4,)不是表示4的行向量，而看成一个list，共4个元素，存在shape[0]中

而数学中的行向量和列向量是

```python
[[1],[2],[3],[4]]. 
[[1,2,3,4]]
利用reshape可以转换
b.reshape(1, 4)
b.reshape(4, 1)
所以(4,)并不代表(4,1), 
那么在广播时，将4当成最后一个维度去匹配，如果匹配上才可以广播.
```

![image-20240208163921388](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208163921388.png)

![image-20240208164123912](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208164123912.png)

numpy的前向传播实现，并没有转化为矩阵，因为只有TensorFlow考虑了转化为矩阵，提高效率。

这种实现是朴素实现，但是我们要的是更简化的实现。

对于矩阵截取，如果有一个维度变成0，就会变成list，而不是[[]]形式。

当变成向量时，利用dot对应项相乘相加变为一个数。

##### 难点理解

![image-20240208170516604](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240208170516604.png)

##### 解释

是的，由于掩层的存在，部分神经元的输出会被置为零，从而只保留了剩余神经元的权重（w）进行计算。这导致每个神经元的权重参数可能是不同的，因为它们不同神经元的连接方式和强度会受到掩层的随机性影响。

掩层的作用是在训练过程中减少神经元之间的依赖关系，以防止过拟合。通过随机丢弃一部分神经元的输出，网络被迫学习更加鲁棒和独立的特征表示。这样，在训练过程中，每个神经元的权重会根据保留的神经元的输出进行更新，而被置为零的神经元的权重不会得到更新。

因此，由于掩层的随机性质，每次训练迭代都会得到略微不同的权重参数。然而，随着训练的进行，网络的参数会逐渐收敛到一个相对稳定的状态，这些参数在整个训练集上能够产生较好的预测结果。最终训练出的模型的权重参数可能会略有不同，但它们应该在一定范围内相似，并且能够对新的未见过的数据进行准确的预测。

矩阵优化

![image-20240209140209337](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240209140209337.png)

##### numpy常见的乘法总结

一维向量在广播时，只会以行向量去广播。

![image-20240209143828069](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240209143828069.png)

如果可以转化为列向量那么就变成 3 * 1，那么就可以广播。

![image-20240209143857106](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240209143857106.png)

所以在广播中，把他认为行向量。

在matmul中，发现行向量可以完成广播，然后矩阵乘法。

![image-20240209144031210](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240209144031210.png)

当一维向量变成第二个参数，只能当作列向量使用。

![image-20240209144712938](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240209144712938.png)

如果我们把第二个维度的一维向量，看做行向量就会报错

![image-20240209144847236](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240209144847236.png)

再次验证，

可以看到作为第一个参数时，就没有报错

![image-20240209145006335](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240209145006335.png)

那么我们假设他为列向量，可以看到不匹配

![image-20240209145104669](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240209145104669.png)

dot同理。

a @ b同理。

广播不同理。

```python
对于matmul，dot，@. 一维向量在前只能看做行向量[a,b,c]，一维向量在后只能看做列向量。[a
                                                                         b
                                                                         c]
对于广播，只看作行向量广播 即 1行 * k 列
其他运算普遍按照行向量进行即可，有特例再记住。
```

