##### 特征缩放

当一个指标的取值范围相对于其他指标很大时，一个好的模型，更可能学会选择一个相对较小的参数。

  ![image-20240203123533721](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203123533721.png)

![image-20240203125451932](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203125451932.png)

![image-20240203130039957](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203130039957.png)

###### 平均归一化

![image-20240203131748368](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203131748368.png)

###### 最大值归一化

##### 标准化随机变量归一化

$X -(\mu, \sigma)的分布，不一定是正态$

那么$X-\mu / \sigma$ - (0 , 1), 均值为0，标准差为1。

这个不只是正态分布适合这个变化，任何随机变量都可以完成。

![image-20240203134553907](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203134553907.png)

![image-20240203134124678](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203134124678.png)



![2a56ebc126112541decf4b060e8a6d34](C:\Users\86186\AppData\Roaming\Tencent\QQ\Temp\2a56ebc126112541decf4b060e8a6d34.jpg)

其实误差函数最后还有交互项比如

不去考虑他们，

$w_1w_2 但\\w_2 * w_2<= w_1 * w_2 <= w_1 * w_1$

将其无论抽象为$w_1w_1$ 或  $w_2w_2$ 对系数影响不大。

交互项，当其范围

$有 w_1^2 = k * w_2^2 + b，b为定值. 绘制图形.$

我们将$w_1的系数重整为1，然后构成形如 w_1^2 = k * w_2^2 + b 的式子，其中b为常数，会得到如下图形。$

![image-20240206121633466](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240206121633466.png)

$w_1^2 + 50000w_2^2 + 常数 = 误差值 \space满足这样的w_1^2 和 w_2^2就在一圈等值线上。我们令b = 误差值-常数. 有w_1^2 = -50000w_2^2 + b \\$

可见无论b

为何值，都不影响k，那么k是个极大值，我们可以看到$w_2变化了很小,w_1变化了很多$

那么对于任意误差值，就构成了任意等值圈，都满足$w_1变化很多，而w_2几乎不变。$

所以就构成了一个个细长椭圆。
$$
对于某个误差值，可以看到w_1的系数很小，w_2的系数很大，那么w_2变化0.0001就会让结果产生500的波动，w_1变化500，才会让结果产生250的波动，那么对于等值处。假设某个误差，对应的w_1的范围是0.0001-0.0005,那么每当w_1减少0.0001时，w_2可能要增加500，所以一一映射下来，w_1的范围是0.0001-0.0005可能导致取同等误差时，w_2从500-50000，那么一个变化极小，一个变化极大。会导致椭圆细长，造成一些麻烦。
$$

![image-20240204121722018](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240204121722018.png)













特征缩放，当数据范围相差不大时，不需要使用。

对于变化过大的数据，对其应用归一化技术，变成归一化的数据，相当于建立了一个映射，将原始数据映射为另一组数据，然后求得另一组数据就能得到预测值。

当我们输入这个指标时，进行预测。如果我们建立模型时，对这个指标归一化了，那么建立的模型是归一化后的数据和其他指标的关系。所以我们的输入这个指标也要转化为归一化。

![image-20240203134441445](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203134441445.png)

当特征缩放后
$$
x_1 和 x_2的范围相近.
那么最后的误差式子大致为
w_1^2 + w_2^2 + b = 0
那么w_1变化一定的范围，对应w_2变化一定的范围.
它们变化幅度基本一致。所以会呈现圆形。
$$


##### 如何判断正确收敛

> 画图

![image-20240203135405407](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203135405407.png)

迭代次数与损失函数的图像，呈现收敛。

> 设置一个阈值，当成本j的减少量小于这个阈值就停止训练。

python存在浮点数，可能会无限迭代。

##### 特征工程

转化和结合原始特征

特征工程简单讲就是发现对因变量y有明显影响作用的特征，通常称自变量x为特征，特征工程的目的是发现重要特征

![image-20240203141506001](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203141506001.png)

##### 分类问题

线性回归不能很好地解决分类问题

![image-20240203142836575](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203142836575.png)

线性回归的蓝色实线，求得到了误差最小的情况。选择一个阈值，低于0.5分为良性，高于0.5分为恶性。显然这几个样本都正确分类了。以0.5为阈值，画一条垂直线，那么在垂直线左侧的x，就会低于0.5为良性。在垂直线右侧的x，就会高于0.5对应恶性。对训练集进行了很好的分类。





![image-20240203143346047](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203143346047.png)

同理，绿色线是加入了右侧的一个恶性样本，我们可以看到![image-20240203143404867](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203143404867.png) 

有两个恶性样本位于垂直绿线左侧，被划分为良性。

当样本中出现更大的恶性肿瘤，那么回归拟合的曲线为了减少总体的误差，会

向右移动那么垂直线会继续右移，有更多的恶性样本被分为良性。

##### 逻辑回归解决

![image-20240203144553053](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203144553053.png)

核心函数![image-20240203144602967](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203144602967.png)

![image-20240203144850780](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203144850780.png)

通过调节z，来让

<img src="C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203150112713.png" alt="image-20240203150112713" style="zoom:150%;" />

得到一个理想的结果，而不是让z本身有一个理想的结果。

![image-20240203150435315](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203150435315.png)

这条曲线很好地拟合了样本，达到了最小误差，但是根据0.5为阈值我们发现有恶性分到了良性，其实有更好的曲线

![image-20240203150602175](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203150602175.png)

显然紫色线是最好的，那么我们通过建立 z = wx + b的关系，然后带入每个值得到![image-20240203150647834](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203150647834.png)

然后根据![image-20240203150653778](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203150653778.png)决定参数变化。

##### 边界决策

![image-20240203151123155](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203151123155.png)

那么就有

$wx + b >= 0$ 反映在图上就是$一条直线左侧的点会被分为0，右侧的点会被分为1$

紫色线就是决策边界

可以理解成指标的关系图，二维可以表示两个指标的关系。三维可以表示三个指标的关系。

> $z = wx + b, w=1,b=3. 即z=x+3,当z为0作为分界线，即x = -3,所以可以知道通过x = -3来区分正类和负类$
>
> 然后通过带入z = x + 3, 可以得到函数值，然后带入到概率函数，可以得到对应的概率。 那么代价函数最后会让所有正类尽量被正确
>
> 区分，并且z得到的概率比较大。
>
> 所以下面的图，只是让z=0，进行降维，让我们进行正负类的区分。但是具体的值，还要将点带入z，然后带入概率函数才能得到，而
>
> 不是简单的距离分界线的远近。
>
> 比如 z = 2*x + 6 和 z = x + 3 分界线都在图中是-3，都对正类负类产生了正确的区分，但是产生的代价不同。 因此下面的图，只是去
>
> 看分类结果，而没有考虑代价，所以达到降低维度。
>
> z = x + 3，当z = 0 ，x =-3， 因为线性函数，在-3左右两侧一定会有要么左侧全>0,要么右侧全>0.反之亦然。

![img](file:///C:\Users\86186\Documents\Tencent Files\969118579\Image\C2C\833d3e753baf33597349e0720e3a07eb.jpg)
$$
z = wx + b,只考虑分界线就可以完成降低维度。 比如我们只考虑 wx + b = 0, 因为wx + b单调，所以两侧一遍大0，一边小于0.
 即 x = - b/w , 那么当x一定时，-b/w 一定。 通过调节w,b. 使得损失函数最小。
 损失函数可以理解成概率，正类概率为1，负类概率为0.那么负类代入wx+b后得到<0的值，会产生小偏差。正类带入得到>0的值，会产生小偏差。
 所以正类在 wx + b = 0 的一侧，负类在 wx + b=0的另一侧。那么通过条件 w,b. 让正类代入后，正值更大。负类代入后，负值更大。这样就得到最小的误差，即最优的函数。
$$




![image-20240203151818861](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203151818861.png)

![image-20240203152149234](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203152149234.png)

##### 如何衡量最优参数

逻辑回归的代价函数如果选择

![image-20240203154652479](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203154652479.png)

f就是我们的![image-20240203154710721](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203154710721.png)

但是代入f，因为我们的f不是线性表达式，得到的代价函数如图。

![image-20240203154810731](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203154810731.png)

使用梯度下降会陷入局部最小



![image-20240203163133009](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203163133009.png)

![image-20240203163301382](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203163301382.png)

将

![image-20240203163311164](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203163311164.png)

转化为

![image-20240203163357813](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203163357813.png)

也同样评判了预测误差，经过这样的转化，那么可以验证出总体的损失函数变成了凸函数。

由此可以梯度下降，来求最小值，不会陷入局部最小。这个转化十分巧妙，属于数学领域的知识。只需要知道转

化后，依旧能够反映出预测值和真实值之间的误差，尽管数值和以前不同，但是我们评判误差要的是相对关系，

只要保证预测正确的误差小，预测错误的误差大，预测正确的比预测错误产生的误差小，只要满足这样的关系就

可以任意转化，不必局限于均差平方，但是这个转换的目的，是转换后整体的损失函数变成了一个凸函数，我们

不会陷入局部最小值，我们可以利用梯度下降进行求解。

当不是二分类问题时，用推广式，转化为凸函数

![image-20240203164053243](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203164053243.png)

这个代价函数是大量实验找出的，比较好的

自己计算后，我们得到他的导数存在这样的关系（先化简，再求导，搞清楚谁是变量）

![image-20240203171712786](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240203171712786.png)

然后就可以带入，进行梯度下降求解。