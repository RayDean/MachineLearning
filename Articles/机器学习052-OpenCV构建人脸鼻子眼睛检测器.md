【火炉炼AI】机器学习052-OpenCV构建人脸鼻子眼睛检测器
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2，opencv-python 3.4.2)

有两个重要的概念需要澄清一下：人脸检测：是指检测图像或视频中是否存在人脸，以及定位人脸的具体位置，人脸识别：确定图像或视频中的人脸是张三还是李四还是其他某人。故而人脸检测是人脸识别的基础和前提条件。

在这一章我们来学习如何用OpenCV构建人脸检测器，鼻子检测器和眼睛检测器。

<br/>

## 1. 构建人脸检测器

前面提到过，人脸检测器是确定图像中人脸位置的过程，我们将用Haar级联来构建人脸检测器。Haar级联通过在多个尺度上从图像中提取大量的简单特征来实现，这些简单特征包括有边，角，线，矩形特征等，然后通过创建一系列简单的分类器来做训练。

Haar级联是一个基于Haar特征的级联分类器，所谓级联分类器，是把多个弱分类器串联成一个强分类器的过程，弱分类器是指性能受限，预测准确度不太高的分类器，所以此处的串联实际上就是机器学习中的Boost方法，即集成方法。所以Haar分类器 =  Haar-like特征 + 积分图方法 + AdaBoost + 级联。关于Haar-like特征和积分图的概念，可以参考博文：[浅析人脸检测之Haar分类器方法](http://www.cnblogs.com/ello/archive/2012/04/28/2475419.html).

### 1.1 对单张图片进行人脸检测 

```py
# 构建单张图片的人脸检测器
def img_face_detector(img_path,face_cascade_file):
    image=cv2.imread(img_path)
    face_cascade=cv2.CascadeClassifier(face_cascade_file)
    if face_cascade.empty(): 
        raise IOError('Unable to load the face cascade classifier xml file!')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face_rects=face_cascade.detectMultiScale(gray,1.3,5)
    # 在检测到的脸部周围画矩形框
    for (x,y,w,h) in face_rects:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
    return image
```

```py
# 测试一下这个人脸检测器：
image1=img_face_detector('E:\PyProjects\DataSet\FireAI/face1.jpg',
                         'E:\PyProjects\DataSet\FireAI\cascade_files/haarcascade_frontalface_alt.xml')
image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
plt.imshow(image1)
```
通过对下面几张图片进行人脸检测，得到的结果分别为：

![](https://i.imgur.com/dOM3Hon.png)

![](https://i.imgur.com/iIWx3pa.png)

![](https://i.imgur.com/jgYHWwF.png)


### 1.2 对视频流进行人脸检测 

视频流的本质其实就是图片，将图片按照一定的每秒帧率fps播放出来即可。故而我们在对视频进行分析时，需要从视频流中捕获图片，对图片进行分析。

```py
# 对视频流进行人脸检测
def video_face_detector(face_cascade_file):
    face_cascade=cv2.CascadeClassifier(face_cascade_file)
    if face_cascade.empty(): 
        raise IOError('Unable to load the face cascade classifier xml file!')
    capture=cv2.VideoCapture(0)
    
    while True:
        _,frame=capture.read() # 捕获当前帧
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        # 在检测到的脸部周围画矩形框
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        
        cv2.imshow('Video Face Detector', frame)
        key=cv2.waitKey(1) # 按ESC退出检测
        if key==27:
            break
    capture.release()
    cv2.destroyAllWindows()
```

<br/>

## 2. 构建鼻子检测器

### 2.1 对单张图片进行鼻子检测

```py
# 构建单张图片的鼻子检测器
def img_nose_detector(img_path,face_cascade_file,nose_cascade_file,show_face=True):
    image=cv2.imread(img_path)
    face_cascade=cv2.CascadeClassifier(face_cascade_file)
    if face_cascade.empty(): 
        raise IOError('Unable to load the face cascade classifier xml file!')
    nose_cascade=cv2.CascadeClassifier(nose_cascade_file)
    if nose_cascade.empty(): 
        raise IOError('Unable to load the nose cascade classifier xml file!')
        
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face_rects=face_cascade.detectMultiScale(gray,1.3,5)
    # 在检测到的脸部周围画矩形框
    for (x,y,w,h) in face_rects:
        if show_face: cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
        roi=gray[y:y+h,x:x+w]
        nose_rects=nose_cascade.detectMultiScale(roi,1.3,5)
        for (x_nose,y_nose,w_nose,h_nose) in nose_rects:
            cv2.rectangle(image,(x+x_nose,y+y_nose),(x+x_nose+w_nose,y+y_nose+h_nose),
                         (0,255,0),3)
            break # 一张脸上只能有一个鼻子，故而此处break
    return image
```

![](https://i.imgur.com/3EVqtBp.png)

### 2.2 对视频流进行鼻子检测 

同样的，对视频流进行鼻子检测的代码为：

```py
# 对视频流进行鼻子检测
def video_nose_detector(face_cascade_file,nose_cascade_file):
    face_cascade=cv2.CascadeClassifier(face_cascade_file)
    if face_cascade.empty(): 
        raise IOError('Unable to load the face cascade classifier xml file!')
    nose_cascade=cv2.CascadeClassifier(nose_cascade_file)
    if nose_cascade.empty(): 
        raise IOError('Unable to load the nose cascade classifier xml file!')
        
    capture=cv2.VideoCapture(0)
    
    while True:
        _,frame=capture.read() # 捕获当前帧
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray)
        # 在检测到的脸部周围画矩形框
        for (x,y,w,h) in face_rects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            roi=gray[y:y+h,x:x+w]
            nose_rects=nose_cascade.detectMultiScale(roi,1.3,5)
            for (x_nose,y_nose,w_nose,h_nose) in nose_rects:
                cv2.rectangle(frame,(x+x_nose,y+y_nose),(x+x_nose+w_nose,y+y_nose+h_nose),
                             (0,255,0),3)
                break # 一张脸上只能有一个鼻子，故而此处break
        cv2.imshow('Video Face Detector', frame)
        key=cv2.waitKey(1) # 按ESC退出检测
        if key==27:
            break
    capture.release()
    cv2.destroyAllWindows()
```

同样的，可以构建对单张图片和视频流的眼睛检测器，具体代码可以看（[**我的github**](https://github.com/RayDean/MachineLearning)）

![](https://i.imgur.com/vDHz2Tj.png)


当然，还可以建立函数同时对鼻子和眼睛进行检测，只需要对原来的鼻子检测函数做少许修改即可。


**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1，此处使用Haar级联构建了人脸，鼻子，眼睛检测器，能够很好的检测到图片，视频流中的各个结构信息。**

**2，从结果上可以看出，虽然能够有效检测，但是还有些人脸或鼻子，眼睛等难以被检测到，此时可能要调整检测函数detectMultiScale()的参数，如果调整参数仍然不理想，就需要修改特征检测级联文件cascade_file这个xml了。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**


<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/MachineLearning)）上，欢迎下载。

参考资料:

1, Python机器学习经典实例，Prateek Joshi著，陶俊杰，陈小莉译