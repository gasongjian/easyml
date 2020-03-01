# EasyML 

## 简介

EasyML 模块的核心理念是把数据科学能模块化的部分抽象出来，用于简化并规范算法工程师的工作。


我将机器学习的相关工作抽象成四大块: 

- 数据校验(check)：描述并校验特征本身、特征之间、特征与目标变量 的分布或相关关系；
- 模型评估(eval)：刻画模型的效果（指标、分布、可视化、泛化性能等）
- 模型解释(explain)：理解在单个/多个样本中单个/多个特征是如何作用于模型的;
- 学习日志(logger)：记录一次项目中每一次调试的过程与结果，方便查找和复现；
- 模型应用(model)：记录并将模型文件分发成各种格式；

## TODO

希望能解决的一些问题

1. 寻找一种通用的方式来抽象特征；
2. 尝试度量特征工程后的效果；
3. 尝试绘制模型的决策边界，尝试可视化模型的决策过程；
4. 方便工程师自定义；


## 参考

- pandas-profiling 包
- [决策边界可视化](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html#sphx-glr-auto-examples-ensemble-plot-forest-iris-py)
