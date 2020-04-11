# machine-learning
经典机器学习算法python实现

## KNN
- [x] knn回归算法
- [ ] kd-tree算法实现

## Linear Regression
- [x] 求解正规方程
- [x] 批量梯度下降法
- [x] 随机梯度下降法
- [ ] 小批量梯度下降法

## PCA
- 梯度上升法实现PCA，在特征比较多时会出现最后几个主成分计算错误的情况，可能是因为后期损失函数过于平坦，导致梯度下降法无法收敛到全局最优
- [ ] PCA 实现 `explained_variance_ratio_`
- [ ] SVD分解实现PCA

## 偏差与方差
- 解决高方差的办法
    - 降低模型复杂度
    - 减少数据维度；降噪
    - 增加样本数
    - 使用验证集
    - 模型正则化
- 模型正则化
    - 惩罚函数不用考虑 $\theta_0$

## Logistics Regression
- 多分类问题 OvO OvR
- 可以利用 `decision_function` 来改变预测的阈值

## 模型评价
- 对于skewed data，准确率不能完全说明算法的性能，需要精准率和召回率。
- 在不同应用场景中精准率和召回率的重要性不同，比如在股票预测中精准率更重要，而在癌症检查中召回率更重要，实际情况中precision和recall需要权衡取舍，或者取得一个平衡
- [] 多分类中的混淆矩阵和ROC曲线

## SVM
- SVM 使用之前需要对数据进行归一化