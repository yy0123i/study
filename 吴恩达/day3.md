##### 过拟合

>  增加更多的数据，当样本点很少拟合出的波动很大，但样本点足够多，波动也就变小了。

![image-20240205165625218](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205165625218.png)

> 减少指标数量
>
> 但是如何选择指标，又是一个新的问题。

![image-20240205165815676](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205165815676.png)

> 正则化技术

![image-20240205170049304](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205170049304.png)

不是完全消灭一个指标，即让参数为0.而是让他为一个较小值。

一般不鼓励b减小

###### 实现

![image-20240205170659794](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205170659794.png)

假如惩罚项，因为在损失函数后面给$w_3和w_4加了一个很大的系数，那么想让损失函数收缩，需要保证w_3和w_4足够小。$

因为一般我们并不知道应该对谁乘法所以我们直接对所有都加一个惩罚，也会得到一个比较光滑的曲线

![image-20240205171205571](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205171205571.png)

除以2m，是防止训练样本增大对模型造成影响。

$如果\lambda过大，会导致惩罚项太大，那么每个参数趋于0，那么模型会过于简单，类似于只有b在作用，会欠拟合。$

$如果\lambda过小，会导致惩罚项太小，那么每个参数依旧很大，那么模型会过于合身，无法解决过拟合。$

##### 正则化线性回归

带惩罚项的导数

![image-20240205174924397](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205174924397.png)

![image-20240205175101214](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205175101214.png)

之前的是

$w_j = w_j - 导数项$

$w_j = w_j *(1-\alpha\lambda/m) - 导数项$  

这样实现了每次让$w_j缩小一点$

##### 正则化逻辑回归

![image-20240205175438790](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205175438790.png)

![image-20240205175650865](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205175650865.png)

##### 神经网络的兴起

![image-20240205182028994](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240205182028994.png)