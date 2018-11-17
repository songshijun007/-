'''
本文件功能为计算香农熵
'''
from math import log
"""
函数说明:创建测试数据集
Returns:
    data - 数据集
    labels - 分类属性
"""
def creatData():
    data =  [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']   #分类属性 最后一列为是否放贷
    return data,labels
"""
函数说明:计算给定数据集的经验熵(香农熵)
    data - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""


def calcShannonEnt(data):
    num = len(data)  # 返回数据集的行数
    labelscount = {}  # 保存每个标签出现的次数
    for feature in data:  # 对每组特征向量进行统计
        currentLabel = feature[-1]  # 提取标签(Label)信息
        if currentLabel not in labelscount.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelscount[currentLabel] = 0
        labelscount[currentLabel] += 1  # Label计数
    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelscount:  # 计算香农熵
        prob = float(labelscount[key]) / num  # 选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannonEnt  # 返回经验熵(香农熵)


if __name__ == '__main__':
    data, features = creatData()
    print(data)
    print(calcShannonEnt(data))
