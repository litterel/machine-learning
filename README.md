# machine-learning
经典机器学习算法python实现
- KNN
    - [x] knn回归算法
    - [ ] kd-tree算法实现

- Linear Regression
    - [x] 求解正规方程
    - [x] 批量梯度下降法
    - [x] 随机梯度下降法
    - [ ] 小批量梯度下降法

- PCA
    - 梯度上升法实现PCA，在特征比较多时会出现最后几个主成分计算错误的情况，可能是因为后期损失函数过于平坦，导致梯度下降法无法收敛到全局最优
    - [ ] PCA explained_variance_ratio_
    - [ ] SVD分解实现PCA

- 偏差与方差
    - 解决高方差的办法
        - 降低模型复杂度
        - 减少数据维度；降噪
        - 增加样本数
        - 使用验证集
        - 模型正则化
    - 模型正则化
        - 惩罚函数不用考虑 $\theta_0$