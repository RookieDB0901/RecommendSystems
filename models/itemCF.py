"""
数据集；基于GroupLens提供的MovieLens的其中一个小数据集ml-latest-small。该数据集包含700个用户对带有6100个标签的10000部电影的100000条评分。
任务：预测用户会不会对某部电影评分，向用户推荐评分可能性最高的电影。
方法：ItemCF。向用户推荐与其历史交互物品相似度总和最高的电影。电影相似度根据历史数据计算。
"""
import random
from tqdm import tqdm
import math
import time
class Dataset():

    def __init__(self, fp):
        self.data = self.loadData(fp)

    def loadData(self, fp):
        """
        :param fp: 数据文件路径
        :return: data: [(user1, item1), (user1, item2), ...]
        """
        data = []
        for line in open(fp, "r"):
            data.append(tuple(map(int, line.strip().split("::")[:2])))
        return data

    def splitData(self, M, k, seed=1):
        """
        :param M: 实验的次数
        :param k: 第几次实验，k∈[0,M)
        :param seed: 随机种子
        :return: train, test
        """
        train, test = [], []
        random.seed(seed)
        # 随机选取1/M的数据作为测试组
        for user, item in self.data:
            if random.randint(0, M-1) == k:
                test.append((user, item))
            else:
                train.append((user, item))

        # 处理成字典的形式
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)

            # data_dict格式为{user: [item1, item2, ...]}
            data_dict = {user: list(data_dict[user]) for user in data_dict}
            return data_dict

        return convert_dict(train), convert_dict(test)

class Metric():
    """
    评价指标：
    1.precision
    2.recall
    """
    def __init__(self, train, test, GetRecommendation):
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()  # 推荐列表 {user: [movie1, movie2, ...]}

    # 为test中的每个用户做推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            rank = self.GetRecommendation(user)
            recs[user] = rank
        return recs

    def precision(self):
        total, hit = 0, 0
        for user in self.test:
            items = set(self.test[user])
            rank = self.recs[user]
            for item, _ in rank:
                if item in items:
                    hit += 1
                total += 1
        return round(hit / total * 100, 2)

    def recall(self):
        total, hit = 0, 0
        for user in self.test:
            items = set(self.test[user])
            rank = self.recs[user]
            for item, _ in rank:
                if item in items:
                    hit += 1
            total += len(items)
        return round(hit / total * 100, 2)

    def eval(self):
        metrics = {
            "Precision": self.precision(),
            "Recall": self.recall()
        }
        print("Metrics:", metrics)
        return metrics

def ItemCF(train, N, K):
    """
    向用户推荐与其历史交互物品相似度总和最高的物品
    :param train: 训练数据集
    :param N: 选取相似度最高的N个物品
    :param K: 推荐最终得分最高的K个物品
    :return:
    """
    # 计算相似度矩阵
    sim = {}  # 相似度矩阵
    num = {}  # 统计每个电影的评分人数
    # 以下循环中，sim[u][v]统计了同时评价过u、v的人数，也等于u、v物品向量的内积
    for user, items in train.items():
        for i in range(len(items)):
            u = items[i]
            if u not in num:
                num[u] = 0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(items)):
                if i != j:
                    v = items[j]
                    if v not in sim[u]:
                        sim[u][v] = 0
                    sim[u][v] += 1
    # 以上sim矩阵存储了两两物品向量的内积，以下再除以向量长度之积得到余弦相似度
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    # 按相似度排序
    sorted_sim = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in sim.items()}

    # 获取推荐列表
    def GetRecommendation(user):
        items = {}  # 记录TopN相似度物品
        seen_items = set(train[user])  # 用户已评论过的物品，将不会再推荐
        for item in train[user]:
            for u, _ in sorted_sim[item][:N]:  # 遍历排序后相似度矩阵的前N个物品，即TopN相似度物品
                if u not in seen_items:  # 用户未评论过该物品
                    if u not in items:
                        items[u] = 0
                    items[u] += sim[item][u]  # 累计物品得分，权重为相似度
        # 取得分最高的K个物品
        recs = list(sorted(items.items(), key=lambda x: x[1], reverse=True))[:K]
        return recs
    return GetRecommendation  # ItemCF返回值为GetRecommendation函数


class Experiment():

    def __init__(self, M, N, K, fp):
        """
        :param M: 实验次数
        :param N: 每个历史物品只取相似度最高的N个物品计算得分，降低计算量
        :param K: 最终获得K个推荐
        :param fp: 数据文件路径
        """
        self.M = M
        self.N = N
        self.K = K
        self.fp = fp

    def worker(self, train, test):
        """
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        """
        # ItemCF返回值为GetRecommendation函数
        getRecommendation = ItemCF(train, self.N, self.K)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0}
        dataset = Dataset(self.fp)
        for i in range(self.M):  # 进行M次实验
            train, test = dataset.splitData(self.M, i)
            print('Experiment {}:'.format(i))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}  # 累加评价指标
        metrics = {k: metrics[k] / self.M for k in metrics}  # 求M次平均
        print('Average Result (M={}, N={}, K={}): {}'.format(self.M, self.N, self.K, metrics))

fp = "./dataset/ml-1m/ratings.dat"
M, K = 8, 10
for N in [5, 20, 40, 160]:
    CF = Experiment(M, N, K, fp)
    CF.run()





