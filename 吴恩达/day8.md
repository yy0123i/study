##### 卷积

###### 步幅

横向移动某个距离后，横向移动到头。纵向移动步幅数，然后再横向移动。

每次卷积会导致图像缩小，并且边角信息只被利用了一次。

###### padding

![image-20240222125325138](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222125325138.png)

![image-20240222125553695](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222125553695.png)

= ${n + 2p - (f - 1)}$

只考虑核的第一列，那么只能在 n + 2p - (f - 1) 移动，即整个核只经过了 n + 2p - (f - 1).

步长为1的情况下。

若2p-f+1 = 0，那么卷积后还是n * n，不会导致图像缩小。p是padding.

当步长为s时，第一行每次可以移动s，最多移动到

首先s <= f，如果s > f， 会有元素丢失这样做没有意义.

当 s <= f， n + 2p - f,  最后剩下f列.

如果n +2p - f / s 没有余数，这时核的尾边贴着n + 2p - f + (f - s) = n + 2p - s,后面正好移动一个s 。想不明白可以看图。

![image-20240222135308532](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222135308532.png)

当(n + 2p - f) / s 有余数，那么余数一定 <= s - 1，那么也就是后面所需的长度最少是 f - (s - 1) = f - s + 1,  那么再移动s步，消耗后为 f - (f - (s - 1)) =  s - 1, s - 1 <= s，后面的距离不能再移动一格，所以最多移动一格。消耗最多的是余数为1，后面需要f - 1格子，显然f满足，移动一个格子后，消耗为1, 1 < s， 显然不能再次移动。而余数为0，即后面需要消耗f个，刚好用完。

即

![image-20240222130115276](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222130115276.png)

###### 将第二个翻转然后做卷积就对了

![image-20240222145450684](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222145450684.png)

######   区别

![image-20240222151913624](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222151913624.png)

输入是一个长方体，厚度每一层是一个通道。

###### filter

filter中的核大小应当一样大，每次卷积与filter的所有核分别作卷积，得到filter相同层数的特征图。

padding 用来解决边界特征与核心特征提取次数不同的问题，让边界点也同样重要。



###### 卷积核个数

fiter的层数。



###### 参数共享

![image-20240222160602954](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222160602954.png)

每个卷积核的参数应用于整张图像，而不是每个区域一个卷积核，这样参数共享可以保证所用的总参数很少，训练起来更方便。

![image-20240222160727637](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222160727637.png)

###### 池化层

卷积层尽可能丰富得提取特征，但是提取的特征并一定有用，而池化层对卷积层提取的特征作压缩，就可以让特征减少，保留最有用的特征。

压缩特征，下采样。

![image-20240222161031426](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222161031426.png)

![image-20240222161329930](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222161329930.png)

在训练过程中，该特征得到的权重大，对应的值大，那么对于结果影响也就越大，也就证明其越重要。

![image-20240222161943182](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222161943182.png)

全连接层再次拉成向量，最后输出5个类别的概率值。

7层神经网络，只有需要更新权重参数的层，才被称为一层。

只有卷积层的卷积核里面有参数，在训练过程中要改变，而 RELU 和 POOL 没有参数改变，只是对卷积层的结果处理。

###### VCC 网络

![image-20240222170010508](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222170010508.png)

pooling层造成的损失，利用特征图的个数来弥补

层数增加后，error反而降低



###### 残差网络

加入新层后，

![image-20240222170315403](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222170315403.png)

![image-20240222170535163](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222170535163.png)

如果加入的一层不好，会被直接跳过。

###### 感受野

当前特征图的某个格子，是由多少个原始数据参与计算得到的。

注意数据堆叠。

![image-20240222172230067](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222172230067.png)

对于中间某个卷积的感受野，可以将其视为一点，然后利用 k - f + 1 = 1, 也可以推出他是由几个点推出的。

然后再往前推，都可以通过目前的 k - f + 1 = 3, 知道 3 * 3 这些网格由多少个上一级数据得来，同理再向前推。

 

反推，一个元素的前一个元素是由n个元素构成，经过3 * 3的卷积。

步长为1，

(k - f  + 1) 带入有 (k - 3 + 1) = 1 得到 k = 3，所以这一个元素由前一个矩阵的3 * 3得来。

同理，

(k - f + 1) 带入有(k - 3 + 1) = 3 得到 k = 5, 所以3 * 3矩阵对应5 * 5的矩阵得来。

同理

k = 7, 所以 5 * 5的矩阵由7 * 7 的原始数据得来，所以最后的一个元素感受野是7

那么步长为1时，感受野大小 (堆叠层数 * 2 + 1)

![image-20240222172827190](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240222172827190.png)

![image-20240223121228766](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223121228766.png)

###### filter和卷积核

一个卷积核解决一个通道，解决完所有通道进行相加，得到特征图一个点。那么这一组卷积核就叫做一个filter。

一个filter对应一个特征图，多通道会进行合成为一个值，每个通道都有一个卷积核。3个通道的卷积核结果相加，得到一个特征图的点。

![image-20240223121708462](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240223121708462.png)

[BatchNorm和LayerNorm——通俗易懂的理解_layernorm和batchnorm-CSDN博客](https://blog.csdn.net/Little_White_9/article/details/123345062)