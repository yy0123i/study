##### 模型

![image-20240202130537679](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202130537679.png)

模型代表一个函数，我们首先选定应用什么类型的模型，假如选择线性模型，那么模型就是

$y = \theta_0x + \theta_1$

模型存在参数，那么训练模型，就是为了得到这个参数。

##### 目的

![image-20240202131341023](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202131341023.png)

选择一组参数，使得我们的模型和训练集标签的误差平方和最小

引入m，是因为如果两个不同的训练集，训练集样本多的得到的值偏大，但是除以样本数量，就与样本数量无关，更加直观的比较不同训练集下的模型的平均误差。（训练误差）

为了将样本数量也考虑进去，我们求每个样本的平均误差。

$1/2m \sum^{m}_{i=1}(h_\theta(x^{(i)} - y^{(i)})) ^ 2$

这里除1/2m的原因是后面求导会消掉1/2，留到后面解决这个疑问。

无论乘什么正数

$1/m \sum^{m}_{i=1}(h_\theta(x^{(i)} - y^{(i)})) ^ 2$ 最小值对应的参数和

k * $1/2m \sum^{m}_{i=1}(h_\theta(x^{(i)} - y^{(i)})) ^ 2$ 最小值对应的参数一致，所以可以作这个变化。

#####  代价函数

![image-20240202133115341](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202133115341.png)

误差函数，用于确定最优参数。误差平方和对于大多数回归问题，都是一个比较好的误差函数。

###### 当固定一个参数时

假设$\theta_0 = 0, 那么绘制出 \theta_1 关于 J(\theta_1) 的图形，寻找最低点就是代价最小的点$

![image-20240202134830159](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202134830159.png)

每一个$\theta_1$ 都对应一条直线，也就是一个模型。

当两个参数都变时，得到的是一个碗形状，最低点对应的$\theta_1,\theta_2$ 就是最优参数

![image-20240202135231710](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202135231710.png)

>  利用等高线图，展示碗型图

![image-20240202135608143](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202135608143.png)

在同一等高线（圆弧）的点，对应的$\theta_1\space \theta_2$ 对应的代价函数值相同，里圈的代价值小。所以最优参数选择在椭圆中心处。

###### 如何快速找到最小值

> 梯度下降法

选取一个起始点，然后根据这个点的任意方向，寻找下降最快的方向增加一个小增量。然后迭代。最后能够趋于一个局部最小值点。

![image-20240202143239849](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202143239849.png)

选取两个不同的起始点，最后趋于的局部最小点可能会不同。

理论基础

![image-20240202143352545](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202143352545.png)

$\alpha * \frac{\part J(\theta_0,\theta_1)}{\part \theta_j}$也就是我们提到的一个小增量，$\alpha$它被称作学习率。

误区

![image-20240202144207347](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202144207347.png)

都应用的是$(\theta_0,\theta_1)$ 这一点的偏导，所以不能在中途先对$\theta_0$ 更新

学习率过小和过大

![image-20240202145318395](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202145318395.png)

在最低点导数为0，那么$\theta_1$ 不会发生改变，也就完成了收敛。 

![image-20240202145513429](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202145513429.png)

梯度下降的增量递减性质，只需要设置好初始学习率，不必考虑靠近最小值时缩小学习率。

![image-20240202150424747](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202150424747.png)

![image-20240202150431442](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202150431442.png)

当接近最小值点时，![image-20240202150445261](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202150445261.png)的值会很小，那么增量也就变小了。可以看到在越靠近最小值，移动的距离越小。

###### 线性回归应用

![image-20240202150928731](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202150928731.png)

梯度下降会陷入局部最优解，因为存在多个局部最小值

![image-20240202151426738](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202151426738.png)

但是对于线性回归的代价函数，都是

![image-20240202151443587](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202151443587.png)

往往存在唯一极小值，也就是最小值。局部最优解等于全局最优解。

难点: 偏导数下降的理解。

方向向量，在$与x轴正向夹角\alpha的方向,移动1的单位距离，x坐标的变化值，y坐标的变化值.$

$换句话说，x，y的变化量决定了移动之后的方向，当变化量是方向向量的k倍，就是朝向方向向量方向移动了k个距离。$  

![image-20240202153336758](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202153336758.png)

这个式子，在数学推导可以严格证明。

一个点可以四面八方

![image-20240202153521816](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202153521816.png)



![image-20240202154106245](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202154106245.png)

偏导其实求的是这两个方向的变化，方向可以用向量表示。

![image-20240202154150846](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202154150846.png)

$(x + k * cos(\alpha),y + k * sin(\alpha))$  那么就说明朝向图的那个方向变化后的点。

![image-20240202154406385](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202154406385.png)

代表这个方向的变化率，那么这个方向的自变量是![image-20240202160115801](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202160115801.png)



考虑方向时，我们考虑投影下去的一圈即可，不伴有高度。那么这个方向变化t，对应x变化tcosa，y变化tsina。那么值变成f(x+tcosa,y+tsina),这样就得到这个方向变化t的变化率。![image-20240202160314524](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202160314524.png)

![image-20240202160532500](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202160532500.png)

由此得到，任意方向变化t，方向导数方向会让函数值变化最大。因此

![image-20240202160642899](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202160642899.png)

学习率相当于t，对于一定的学习率，让函数值变化最大的是两个偏导数的方向。 

$(\part z/\part x, \part z / \part y)$  那么这个方向变化 $\alpha$ 的效果是

![image-20240202162506029](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202162506029.png)



那么改变学习率，可以保证向下降最快的方向下降多远的距离。

##### 代码实现





##### 多元回归

![image-20240202171552307](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202171552307.png)

转化为两个向量的内积，给定指标向量x，参数向量$\theta$ 就可以得到某个样本的预测值。

![image-20240202171938695](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202171938695.png)

完成梯度下降向n元的推广.

![image-20240202172135710](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202172135710.png)

![image-20240202173053248](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202173053248.png)

$每一个残差，都是关于 \theta_{1-m} 的变量$ ，那么这个变量的最大变化值，在向量

$(\part y/\theta_1,\part y/\theta_2,...,\part y/\theta_n)$ 这个方向变化最快

![image-20240202180643900](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202180643900.png)v

![image-20240202180911112](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202180911112.png)

那么这个曲线方程为

$J(\theta) = 1/（2 * m） * \sum_{i=1}^{m} (\theta_1 * x_1^{(i)} + \theta_2 * x_2^{(i)} + \theta_0 - y^{(i)})^2$

当$x_1 的值比较大，x_2的值比较小$

![img](file:///C:\Users\86186\Documents\Tencent Files\969118579\Image\C2C\5612b1de6825d21d7a88a2236daa36a2.jpg)

![image-20240202223203206](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202223203206.png)

观察梯度下降的式子

![image-20240202222720623](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202222720623.png)
$$
前文提到，等值线在\theta_2 方向很长，在\theta_1 方向很短，x_1范围较大。x_2范围较小。
$$
![image-20240202222906959](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240202222906959.png)

$\theta_1 方向很短，但是这个值因为乘x_1，x_1的范围很大，那么就会导致在\theta_1方向发生震荡，变化过大。$

$\theta_2 方向很长，但是这个值因为乘x_2，x_2的范围很小，那么就会导致在\theta_2方向变化过小，需要更多次迭代才能到达中心点。$
