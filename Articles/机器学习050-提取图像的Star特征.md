【火炉炼AI】机器学习050-提取图像的Star特征
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2 )

对于图像的特征点，前面我们讨论过边缘检测方法，Harris角点检测算法等，这些检测算法检测的都是图像的轮廓边缘，而不是内部细节，如果要进一步提取图像内部细节方面的特征，需要用到SIFT特征提取器和Star特征提取器。上一篇我们讲解了SIFT特征提取器，下面我们来介绍Star特征提取器。

在博文[特征点的基本概念和如何找到它们](https://www.cnblogs.com/jsxyhelu/p/7520047.html)中提到，OpenCV中具有以下特征点模型：（这些函数的使用也同样适用于OpenCV-Python）

1、Harris-Shi-Tomasi特征检测器和cv :: GoodFeaturesToTrackDetector

 最常用的角点定义是由哈里斯[Harris88]提出的， 这些角点，被称为哈尔角点，可以被认为是原始的关键点；而后被Shi和Tomasi [Shi94]进行了进一步扩展，后者被证明对于大多数跟踪应用来说是优越的。由于历史原因，在OpenCV中叫做”GoodFeatures";
 
2、简单的blob检测器和cv :: SimpleBlobDetector

 提出“斑点”的概念。斑点本质上没有那么明确的局部化，而是表示可能预期随时间具有一定稳定性的感兴趣区域。

3、FAST特征检测器和cv :: FastFeatureDetector

最初由Rosten和Drummond [Rosten06]提出的FAST（加速段测试的特征），其基本思想是，如果附近的几个点与P类似，那么P将成为一个很好的关键点。

4、SIFT特征检测器和cv :: xfeatures2d :: SIFT

由David Lowe最初于2004年提出的SIFT特征（尺度不变特征变换）[Lowe04]被广泛使用，是许多随后开发的特征的基础；SIFT特征计算花销很大，但是具有高度的表达能力。

5、SURF特征检测器和cv :: xfeatures2d :: SURF

SURF特征（加速鲁棒特征）最初由Bay等人于2006年提出[Bay06，Bay08]，并且在许多方面是我们刚刚讨论的SIFT特征的演变。SURF所产生的特征不仅计算速度快得多，并且在许多情况下，它对SIFT特征观察到的方向或照明变化的鲁棒性也更强。

6、Star / CenSurE特征检测器和cv :: xfeatures2d :: StarDetector

Star特征，也被称为中心环绕极值（或CenSurE）功能，试图解决提供哈尔角点或FAST特征的局部化水平的问题，同时还提供尺度不变性。

7、BRIEF描述符提取器和cv :: BriefDescriptorExtractor

BRIEF，即二进制鲁棒独立基本特征，是一种相对较新的算法，BRIEF不找到关键点；相反，它用于生成可通过任何其他可用的特征检测器算法定位的关键点的描述符。

8、BRISK算法

Leutenegger等人介绍的BRISK40描述符,试图以两种不同的方式改进Brief（Leutenegger11）。 首先，BRISK引入了自己的一个特征检测器（回想一下，Brief只是一种计算描述符的方法）。 其次，BRISK的特征本身虽然与BRIEF原则相似，却尝试以提高整体功能的鲁棒性的方式进行二值比较。

9、ORB特征检测器和cv :: ORB

创建了ORB功能[Rublee11]，其目标是为SIFT或SURF提供更高速的替代品。ORB功能使用非常接近于FAST（我们在本章前面看到的）的关键点检测器，但是使用了基本上不同的描述符，主要基于BRIEF。

10、FREAK描述符提取器和cv :: xfeatures2d :: FREAK

FREAK描述符最初是作为Brief，BRISK和ORB的改进引入的，它是一个生物启发式的描述符，其功能非常类似于BRIEF，主要在于它计算二进制比较的领域的方式[Alahi12]。

11、稠密特征网格和cv :: DenseFeatureDetector类

cv :: DenseFeatureDetector class53的目的只是在图像中的网格中生成一个规则的特征数组。

<br/>

## 1. 使用Star提取图像特征点

具体代码和我上一篇文章很类似，可以先看看[【火炉炼AI】机器学习049-提取图像的SIFT特征点](https://www.cnblogs.com/RayDean/p/9831163.html)

```py
star = cv2.xfeatures2d.StarDetector_create() # 构建Star特征点检测器对象
keypoints = star.detect(gray, None) # 用Star特征点检测器对象检测灰度图中的特征点
```

```py
# 将keypoints绘制到原图中
img_sift = np.copy(img)
cv2.drawKeypoints(img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#画在原图上，第一个是原图，第二个image是目标图，这个flags既包括位置又包括方向，也有只有位置的flags
#圆圈的大小表示特征的重要性大小，主要在桌子腿上，腿上边缘和棱角比较多，更有代表性

# 显示绘制有特征点的图像
plt.figure(12,figsize=(15,30))
plt.subplot(121)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('Raw Img')

plt.subplot(122)
img_sift_rgb=cv2.cvtColor(img_sift,cv2.COLOR_BGR2RGB)
plt.imshow(img_sift_rgb)
plt.title('Img with Star features')
```

![](https://i.imgur.com/1eTRql8.png)


**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1，Star特征点的提取方法在代码上很容易实现，cv2.xfeatures2d.StarDetector_create()这个函数就可以构建Star检测器并用detect方法就可以实现。**

**2，Star特征点提取器和SIFT特征点提取器的结果有很多相似之处，都是在桌腿附近有很多特征点。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/MachineLearning)）上，欢迎下载。

参考资料:

1, Python机器学习经典实例，Prateek Joshi著，陶俊杰，陈小莉译