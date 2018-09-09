【火炉炼AI】机器学习020-使用K-means算法对数据进行聚类分析
-

(本文所使用的Python库和版本号: Python 3.6, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2 )

前面的机器学习类文章（编号从010-019）都是关于**监督学习**，但是从本篇文章开始，炼丹老顽童要开始讲解**无监督学习**方面，无监督学习是指处理的数据没有任何形式的标记，我们没有对训练数据集进行实现的类别划分，故而相当于抹黑处理，要让机器学习自己找出样本所属的类别，那么机器学习通过什么方式来找出“所属类别”了？这就是聚类算法的作用了。

聚类算法，其核心思想就是中国的“人以类聚，物以群分”，就是机器学习将通过一定的算法来将样本划分类别，使得相互之间**相似**的样本划分为一个类别，**不相似**的样本划分为不同的类别中。

K-means算法是最流行的聚类算法之一，这种算法常常利用数据的不同属性将输入数据划分为K组，这种划分是使用最优化的技术实现的，让各组内的数据点与该组中心点的距离平方和最小化。

说起来很抽象，让人感觉云里雾里，那么我们看一个简单的实例吧。

<br/>

## 1. 准备数据集

本次所使用的数据集是我前面的文章[【火炉炼AI】机器学习010-用朴素贝叶斯分类器解决多分类问题](https://juejin.im/post/5b68334ce51d4519115d587e)中所采用的数据集，一个具有四种不同类别，两种不同features的小数据集，其加载方法和显示方法如下所示。

```Python
# 准备数据集
data_path='E:\PyProjects\DataSet\FireAI/data_multivar.txt'
df=pd.read_csv(data_path,header=None)
# print(df.head())
# print(df.info()) # 查看数据信息，确保没有错误
dataset_X,dataset_y=df.iloc[:,:-1],df.iloc[:,-1]
# print(dataset_X.head())
print(dataset_X.info())
print('-'*100)
print(dataset_y.head())
dataset_X=dataset_X.values
dataset_y=dataset_y.values
# print(dataset_X.shape) # (400, 2)
# print(dataset_y.shape) # (400,)
```

**-------------------------------------输---------出--------------------------------**

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 400 entries, 0 to 399
Data columns (total 2 columns):
0    400 non-null float64
1    400 non-null float64
dtypes: float64(2)
memory usage: 6.3 KB
None

**--------------------------------------------完-------------------------------------**

表明结果数据集已经正确地加载到内存中，且每一个features中都没有Null值，我们无需做进一步的缺失值处理。

下面将这个数据集进行2D可视化，如下是可视化的代码：

```Python
# 无标签数据集可视化，将第一列feature作为X，第二列feature作为y
def visual_2D_dataset_dist(dataset):
    '''将二维数据集dataset显示在散点图中'''
    assert dataset.shape[1]==2,'only support dataset with 2 features'
    plt.figure()
    X=dataset[:,0]
    Y=dataset[:,1]
    plt.scatter(X,Y,marker='v',c='g',label='dataset')
    
    X_min,X_max=np.min(X)-1,np.max(X)+1
    Y_min,Y_max=np.min(Y)-1,np.max(Y)+1
    plt.title('dataset distribution')
    plt.xlim(X_min,X_max)
    plt.ylim(Y_min,Y_max)
    plt.xlabel('feature_0')
    plt.ylabel('feature_1')
    plt.legend()
    
visual_2D_dataset_dist(dataset_X)
```

得到的结果如下：

![无标签数据集的可视化](https://i.imgur.com/aEjjlXx.png)

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1. 本数据集的加载很简单，只需用Pandas就可以直接加载，且不需要做其他处理。**

**2. 此处需要注意，无标签数据集的二维平面可视化，不能使用label数据，故而此处的可视化函数和我以往文章中的可视化函数是不一样的，此处需要额外注意。**

**3. 从二维平面散点图中可以看出，这个数据集大概可以分为4个不同的类别，即数据都分布在四个族群里，这就是我们可以用K-mean算法的基础。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

<br/>

## 2. 构建K-means算法

构建K-means算法的过程很简单，和其他的SVM，随机森林算法的构建方式一样，如下代码：

```Python
# 定义一个k-means对象
from sklearn.cluster import KMeans
kmeans=KMeans(init='k-means++',n_clusters=4,n_init=10)
# 这几个参数是初始化设定的，其中n_clusters是从二维散点图中看出大概有4个族群
kmeans.fit(dataset_X)
```

**-------------------------------------输---------出--------------------------------**

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
random_state=None, tol=0.0001, verbose=0)

**--------------------------------------------完-------------------------------------**

虽然此处我们定义了一个KMeans对象，且使用我们的无标签数据集进行了训练，可是训练结果怎么样了？我们怎么知道k-means算法是否正确的划分了不同类别？

所以我们需要一个可视化的结果，就像前面文章中提到的SVM分类结果图一样，此处我们定义了一个专门用于可视化K-means聚类结果的函数，并用该函数来查看此处聚类的效果。代码如下：

```Python
def visual_kmeans_effect(k_means,dataset):
    assert dataset.shape[1]==2,'only support dataset with 2 features'
    X=dataset[:,0]
    Y=dataset[:,1]
    X_min,X_max=np.min(X)-1,np.max(X)+1
    Y_min,Y_max=np.min(Y)-1,np.max(Y)+1
    X_values,Y_values=np.meshgrid(np.arange(X_min,X_max,0.01),
                                  np.arange(Y_min,Y_max,0.01))
    # 预测网格点的标记
    predict_labels=k_means.predict(np.c_[X_values.ravel(),Y_values.ravel()])
    predict_labels=predict_labels.reshape(X_values.shape)
    plt.figure()
    plt.imshow(predict_labels,interpolation='nearest',
               extent=(X_values.min(),X_values.max(),
                       Y_values.min(),Y_values.max()),
               cmap=plt.cm.Paired,
               aspect='auto',
               origin='lower')
    
    # 将数据集绘制到图表中
    plt.scatter(X,Y,marker='v',facecolors='none',edgecolors='k',s=30)
    
    # 将中心点回执到图中
    centroids=k_means.cluster_centers_
    plt.scatter(centroids[:,0],centroids[:,1],marker='o',
                s=100,linewidths=2,color='k',zorder=5,facecolors='b')
    plt.title('K-Means effect graph')
    plt.xlim(X_min,X_max)
    plt.ylim(Y_min,Y_max)
    plt.xlabel('feature_0')
    plt.ylabel('feature_1')
    plt.show()
    
visual_kmeans_effect(kmeans,dataset_X)
```

![K-means聚类对本数据集的聚类结果](https://i.imgur.com/y3rofko.png)

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1. 定义K-means聚类算法的方法很简单，只需要从sklearn.cluster中导入KMeans，并定义一个KMeans对象即可，直接用fit()函数可以直接训练。**

**2. 此处使用k-means聚类算法对数据进行了聚类分析，可以使用函数visual_kmeans_effect()来直接查看聚类后的效果图。**

**3. 虽然可以直观看到效果图，但效果图还是难以量化k-means聚类算法的准确度，这些内容将在后续文章中讲解。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**




<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/MachineLearning)）上，欢迎下载。

参考资料:

1, Python机器学习经典实例，Prateek Joshi著，陶俊杰，陈小莉译