【火炉炼AI】机器学习017-使用GridSearch搜索最佳参数组合
-

(本文所使用的Python库和版本号: Python 3.5, Numpy 1.14, scikit-learn 0.19, matplotlib 2.2 )

在前面的文章（[【火炉炼AI】机器学习012-用随机森林构建汽车评估模型及模型的优化提升方法](https://juejin.im/post/5b684abbf265da0fa759f463)），我们使用了验证曲线来优化模型的超参数，但是使用验证曲线难以同时优化多个参数的取值，只能一个参数一个参数的优化，从而获取每个参数的最优值，但是有时候，一个非常优秀的模型，可能A参数取最优值时，B参数并不一定是最优值，从而使得验证曲线的方式有其自身的弊端。

此处介绍的使用GridSearch来搜索最佳参数组合的方法，可以避免上述弊端，GridSearch可以同时优化多个不同参数的取值。

<br/>

## 1. 准备数据集

数据集的准备工作和文章（[【火炉炼AI】机器学习014-用SVM构建非线性分类模型](https://juejin.im/post/5b69aef6f265da0f82025693)）中一模一样，此处不再赘述。

<br/>

## 2. 使用GridSearch函数来寻找最优参数

使用GridSearch函数来寻找最优参数，需要首先定义要搜索的参数候选值，然后定义模型的评价指标，以此来评价模型的优虐。，GridSearch会自动计算各种参数候选值，从而得到最佳的参数组合，使得评价指标最大化。


```Python
from sklearn import svm, grid_search, cross_validation
from sklearn.metrics import classification_report

parameter_grid = [  {'kernel': ['linear'], 'C': [1, 10, 50, 600]}, # 需要优化的参数及其候选值
                    {'kernel': ['poly'], 'degree': [2, 3]},
                    {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 10, 50, 600]},
                 ]

metrics = ['precision', 'recall_weighted'] # 评价指标好坏的标准

for metric in metrics:
    print("Searching optimal hyperparameters for: {}".format(metric))

    classifier = grid_search.GridSearchCV(svm.SVC(C=1), 
            parameter_grid, cv=5, scoring=metric)
    classifier.fit(train_X, train_y)

    print("\nScores across the parameter grid:") 
    for params, avg_score, _ in classifier.grid_scores_:  # 打印出该参数下的模型得分
        print('{}: avg_scores: {}'.format(params,round(avg_score,3)))

    print("\nHighest scoring parameter set: {}".format(classifier.best_params_))

    y_pred =classifier.predict(test_X) # 此处自动调用最佳参数？？
    print("\nFull performance report:\n {}".format(classification_report(test_y,y_pred)))
```

**-------------------------------------输---------出--------------------------------**

Searching optimal hyperparameters for: precision
Scores across the parameter grid:
{'C': 1, 'kernel': 'linear'}: avg_scores: 0.809
{'C': 10, 'kernel': 'linear'}: avg_scores: 0.809
{'C': 50, 'kernel': 'linear'}: avg_scores: 0.809
{'C': 600, 'kernel': 'linear'}: avg_scores: 0.809
{'degree': 2, 'kernel': 'poly'}: avg_scores: 0.859
{'degree': 3, 'kernel': 'poly'}: avg_scores: 0.852
{'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}: avg_scores: 1.0
{'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}: avg_scores: 0.0
{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}: avg_scores: 0.968
{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}: avg_scores: 0.855
{'C': 50, 'gamma': 0.01, 'kernel': 'rbf'}: avg_scores: 0.946
{'C': 50, 'gamma': 0.001, 'kernel': 'rbf'}: avg_scores: 0.975
{'C': 600, 'gamma': 0.01, 'kernel': 'rbf'}: avg_scores: 0.948
{'C': 600, 'gamma': 0.001, 'kernel': 'rbf'}: avg_scores: 0.968

Highest scoring parameter set: {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}

Full performance report:
precision    recall  f1-score   support

0       0.75      1.00      0.86        36
1       1.00      0.69      0.82        39

avg / total       0.88      0.84      0.84        75

Searching optimal hyperparameters for: recall_weighted

Scores across the parameter grid:
{'C': 1, 'kernel': 'linear'}: avg_scores: 0.653
{'C': 10, 'kernel': 'linear'}: avg_scores: 0.653
{'C': 50, 'kernel': 'linear'}: avg_scores: 0.653
{'C': 600, 'kernel': 'linear'}: avg_scores: 0.653
{'degree': 2, 'kernel': 'poly'}: avg_scores: 0.889
{'degree': 3, 'kernel': 'poly'}: avg_scores: 0.884
{'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}: avg_scores: 0.76
{'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}: avg_scores: 0.507
{'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}: avg_scores: 0.907
{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}: avg_scores: 0.658
{'C': 50, 'gamma': 0.01, 'kernel': 'rbf'}: avg_scores: 0.92
{'C': 50, 'gamma': 0.001, 'kernel': 'rbf'}: avg_scores: 0.72
{'C': 600, 'gamma': 0.01, 'kernel': 'rbf'}: avg_scores: 0.933
{'C': 600, 'gamma': 0.001, 'kernel': 'rbf'}: avg_scores: 0.902

Highest scoring parameter set: {'C': 600, 'gamma': 0.01, 'kernel': 'rbf'}

Full performance report:
precision    recall  f1-score   support

0       1.00      0.92      0.96        36
1       0.93      1.00      0.96        39

avg / total       0.96      0.96      0.96        75

**--------------------------------------------完-------------------------------------**



**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#小\*\*\*\*\*\*\*\*\*\*结\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**

**1. 使用GridSearch中的GridSearchCV可以实现最佳参数组合的搜索，但需要指定候选参数和模型的评价指标。**

**2. 使用classifier.best_params_函数可以直接把最佳的参数组合打印出来，方便以后参数的直接调用**

**3. classifier.predict函数是自动调用最佳的参数组合来预测，从而得到该模型在测试集或训练集上的预测值。**

**\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#**


如果要使用最佳参数来构建SVM模型，可以采用下面的代码来实现：

```Python
best_classifier=svm.SVC(C=600,gamma=0.01,kernel='rbf') # 上面的full performance report的确使用的是最佳参数组合
best_classifier.fit(train_X, train_y)
y_pred =best_classifier.predict(test_X)
print("\nFull performance report:\n {}".format(classification_report(test_y,y_pred)))
```

得到的结果和上面full performance report一模一样。






<br/>

注：本部分代码已经全部上传到（[**我的github**](https://github.com/RayDean/MachineLearning)）上，欢迎下载。

参考资料:

1, Python机器学习经典实例，Prateek Joshi著，陶俊杰，陈小莉译