##### 前向传播

![image-20240219122443764](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240219122443764.png)

在确定参数的情况下，给定输入x，最后可以推断出确定的一个值。神经元由多变少，用于预测。

##### 隐藏层

隐藏层不能使用线性函数，因为这样无论多少个隐藏层，最后得到的都是一个线性函数

![image-20240219171308986](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240219171308986.png)

如果隐层采用线性，那么下一层的输出也变成了线性，无法表示非线性的关系。

##### 反向传播



##### 01数字识别

###### enumerate()

![image-20240219150944388](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240219150944388.png)

生成元组 (index, value)，然后获取 index ，value = next(enumerate)

lst 变为

(0 , 1) , (1, 2) ....

###### 绘制图形

![image-20240219152151268](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240219152151268.png)

![image-20240219153534001](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240219153534001.png)

plt操作的是当前图的axes，当存在多个子图时，利用ax进行操作不同子图。

fig , axes = plt.subplot(几行，几列，当前第几个)

![image-20240219153852201](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240219153852201.png)

plt.imshow() 显示二维数组，变成图像。

![image-20240219155606015](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240219155606015.png)

ax对每一个子图进行操纵。

###### 卷积层，全连接层，池化层

先正向传播，获取到所有的out，然后计算最后的$E_{total}$ ，再反向传递。

忽略最后一层，其它层都是 $out对net求偏导(out*(1-out))再 * 参数$ 就完成了向下一层传递误差。

[机器学习笔记丨神经网络的反向传播原理及过程（图文并茂+浅显易懂）_神经网络反向传播原理-CSDN博客](https://blog.csdn.net/fsfjdtpzus/article/details/106256925)

![image-20240220133758349](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240220133758349.png)

$在net_{o1}处的误差记作\xi_{o1},即对_{o1}求偏导，那么传递给h_1，会求导h_1,那么会乘w_5,所以这样就把误差分配了，按照w_5,w_6分别把误差net_{01}分配给h1,h2$

![image-20240220143912172](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240220143912172.png)=![image-20240220143917215](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240220143917215.png)

然后到达out端，求出net端的误差，就可以再次按照权重分配给上一级，由此完成了误差的传递。

传递好后，利用每个的net端，乘上对对应参数求偏导即可。

这就是误差的传递，以及求出参数的偏导。



利用偏导数，求得成本函数关于参数的偏导。然后根据这个偏导数值，更新参数。

最后所有参数更新，使得成本函数向着下降最快的方向移动，这样就可以完成快速的下降成本。

![image-20240220134207811](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240220134207811.png)

我们将某个参数的误差，看作总误差对其偏导数。那么会得到

$误差_a * 偏导$ 就可以得到其子参数的偏导数。

![image-20240220134601529](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240220134601529.png)

可见，子参数的误差值(误差函数对其偏导)是其父函数的一个倍数.

![image-20240220134927262](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240220134927262.png)

系数依次更新。

![img](file:///C:\Users\86186\Documents\Tencent Files\969118579\Image\C2C\849fac781ded28e834d5cdb7c5dc2fc6.jpg)

![e0a7f8a3d9c5ca846ed66408e018a17d](C:\Users\86186\AppData\Roaming\Tencent\QQ\Temp\e0a7f8a3d9c5ca846ed66408e018a17d.jpg)

隐藏层推导

![image-20240220140726861](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240220140726861.png)

<img src="C:\Users\86186\AppData\Roaming\Tencent\QQ\Temp\ebe980b0b465af3ce6ce26207f1ae9c8.jpg" alt="ebe980b0b465af3ce6ce26207f1ae9c8" style="zoom:33%;" />

