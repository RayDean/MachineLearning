【火炉炼AI】机器学习049-提取图像的SIFT特征点
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2 )

图像中的特征点，就是某一幅图像区别于其他图像的关键点位，在进行这些关键点位的检测时，我们要考虑几个问题，即1，不管怎么旋转目标，要保持目标的特征点不变（即旋转不变性），2，不管这个目标是变大还是变小，其特征点也要保持不变（即尺度不变性），还有比如要求光照不变性等等。

目前对于特征点位的描述有很多种方法和算子，常见的有SIFT特征描述算子、SURF特征描述算子、ORB特征描述算子、HOG特征描述、LBP特征描述以及Harr特征描述。关于这几种算子和特征描述的区别，可以参考博文：[图像特征检测描述(一):SIFT、SURF、ORB、HOG、LBP特征的原理概述及OpenCV代码实现](http://lib.csdn.net/article/opencv/41913)

SIFT特征点，即尺度不变特征变换（Scale-invariant feature transform，SIFT），是用于图像处理领域的一种描述。SIFT特征点在图像处理和计算机视觉领域有着很重要的作用。

SIFT特征点具有很多优点：

1.SIFT特征是图像的局部特征，其对旋转、尺度缩放、亮度变化保持不变性，对视角变化、仿射变换、噪声也保持一定程度的稳定性；

2.区分性好，信息量丰富，适用于在海量特征数据库中进行快速、准确的匹配；

3.多量性，即使少数的几个物体也可以产生大量的SIFT特征向量；

4.高速性，经优化的SIFT匹配算法甚至可以达到实时的要求；

5.可扩展性，可以很方便的与其他形式的特征向量进行联合。

对SIFT特征点的提取主要包括以下四个步骤：

1.尺度空间极值检测：搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。

2.关键点定位：在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。

3.方向确定：基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。

4.关键点描述：在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。

关于SIFT的数学推导和具体含义，可以参考这篇博文：[SIFT特征详解](https://www.cnblogs.com/wangguchangqing/p/4853263.html)

<br/>

## 1. 提取SIFT特征点

### 1.1 安装opencv-contrib-python模块

一般我们使用的是opencv-python模块，但是这个模块中没有xfeatures2d这个方法，因为SIFT算法已经被申请专利，故而从opencv-python中剔除了。

关于这个模块的安装，网上有很多个版本，我也踩了好几个坑，最后发现下面的方法是可用的，先卸载原先的opencv-python模块（如果原先的opencv-python模块的版本号是3.4.2.16，则不需要卸载）。然后安装3.4.2.16这个版本的opencv-python和 opencv-contrib-python即可。

安装方法：

pip install opencv-python==3.4.2.16

pip install opencv-contrib-python==3.4.2.16


### 1.2 提取SIFT特征点

首先构建SIFT特征点检测器对象，然后用这个检测器对象来检测灰度图中的特征点

```py
sift = cv2.xfeatures2d.SIFT_create() # 构建SIFT特征点检测器对象
keypoints = sift.detect(gray, None) # 用SIFT特征点检测器对象检测灰度图中的特征点
```

```py
# 将keypoints绘制到原图中
img_sift = np.copy(img)
cv2.drawKeypoints(img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示绘制有特征点的图像
plt.figure(12,figsize=(15,30))
plt.subplot(121)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Raw Img')

plt.subplot(122)
img_sift_rgb=cv2.cvtColor(img_sift,cv2.COLOR_BGR2RGB)
plt.imshow(img_sift_rgb)
plt.title('Img with SIFT features')
```

![](https://i.imgur.com/ZeQYiGP.png)


**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1, SIFT特征点的提取只需要使用cv2.xfeatures2d.SIFT_create().detect()函数即可，但是要事先安装opencv-contrib-python模块。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/MachineLearning)）上，欢迎下载。

参考资料:

1, Python机器学习经典实例，Prateek Joshi著，陶俊杰，陈小莉译