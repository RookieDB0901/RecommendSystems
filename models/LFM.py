import math
import numpy as np
import random
from utils import Dataset, Metrics
class Dataset():

    def __init__(self, fp):
        self.data = self.loadData(fp)

    def loadData(self, fp):
        data = []
        for line in open(fp, "r"):
            data.append(tuple(map(int, line.strip().split("::")[:2])))
        return data

    def splitData(self, M, K):
        train, test = [], []
        random.seed(1)
        for user, item in self.data:
            if random.randint(0, M-1) == K:
                test.append((user, item))
            else:
                train.append((user, item))

        def convert_dict(data):
            dict_data = {}
            for user, item in data:
                if user not in dict_data:
                    dict_data[user] = set()
                dict_data[user].add(item)
            dict_data = {k: list(v) for k, v in dict_data.items()}
            return dict_data

        return convert_dict(train), convert_dict(test)

class Metrics():

    def __init__(self, train, test, GetRecommendation):
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()

    def getRec(self):
        recs = {}
        for user in self.test:
            recs[user] = self.GetRecommendation(user)
        return recs

    def precision(self):
        total, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            for item, _ in self.recs[user]:
                if item in test_items:
                    hit += 1
                total += 1
        return round(hit / total * 100, 2)

    def recall(self):
        total, hit = 0, 0
        for user in self.test:
            test_items = set(self.test[user])
            for item, _ in self.recs[user]:
                if item in test_items:
                    hit += 1
            total += len(test_items)
        return round(hit / total * 100, 2)

    def eval(self):
        metric = {
            "Precision": self.precision(),
            "Recall": self.recall()
        }
        print("Metric:", metric)
        return metric

def LFM(train, K, N, lr, step, ratio, lmbda):
    """
    :param train: 训练集
    :param K: 隐语义个数
    :param N: 推荐物品个数
    :param lr: 学习率
    :param step: 训练步数
    :param ratio: 采样后负样本数量：正样本数量
    :param lmbda: 正则化系数
    :return: GetRecommendation
    """
    all_items = {}
    for user in train:
        for item in train[user]:
            if item not in all_items:
                all_items[item] = 0
            all_items[item] += 1
    all_items = list(all_items.items())
    items = [x[0] for x in all_items]
    pops = [x[1] for x in all_items]

    def nSamples(data):
        new_data = {}
        for user in data:
            if user not in new_data:
                new_data[user] = {}
            for item in data[user]:
                new_data[user][item] = 1
        for user in new_data:
            seen_items = set(new_data[user])
            num = len(seen_items)
            neg_items = np.random.choice(items, int(ratio * num * 3), pops)
            neg_items = [x for x in neg_items if x not in seen_items][:ratio*num]
            new_data[user].update({x:0 for x in neg_items})
        return new_data

    # P为用户向量矩阵，Q为物品向量矩阵
    P, Q = {}, {}
    for user in train:
        P[user] = np.random.random(K)  # 随机生成K维向量，元素属于[0, 1)
    for item in items:
        Q[item] = np.random.random(K)

    for i in range(step):
        data = nSamples(train)
        for user in data:
            for item in data[user]:
                eui = data[user][item] - (P[user] * Q[item]).sum()
                P[user] += lr * (eui * Q[item] - lmbda * P[user])
                Q[item] += lr * (eui * P[user] - lmbda * Q[item])
        lr *= 0.9  # 调整学习率，一开始收敛更快，后面使得更加精细

    def GetRecommendation(user):
        seen_items = set(train[user])
        recs = {}
        for item in items:
            if item not in seen_items:
                recs[item] = (P[user] * Q[item]).sum()
        recs = list(sorted(recs.items(), key=lambda x: x[1], reverse=True))[:N]
        return recs

    return GetRecommendation


class Experiment():

    def __init__(self, M, N, fp, ratio=1, K=100, lr=0.02, step=100, lmbda=0.01):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: ratio, 正负样本比例
        :params: K, 隐语义个数
        :params: lr, 学习率
        :params: step, 训练步数
        :params: lmbda, 正则化系数
        :params: fp, 数据文件路径
        '''
        self.M = M
        self.K = K
        self.N = N
        self.ratio = ratio
        self.lr = lr
        self.step = step
        self.lmbda = lmbda
        self.fp = fp
        self.LFM = LFM

    # 定义单次实验
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.LFM(train, self.K, self.N, self.lr, self.step, self.ratio, self.lmbda)
        metric = Metrics(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0}
        dataset = Dataset(self.fp)
        for i in range(self.M):
            train, test = dataset.splitData(self.M, i)
            print('Experiment {}:'.format(i))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, ratio={}): {}'.format(self.M, self.N, self.ratio, metrics))

fp = "./dataset/ml-1m/ratings.dat"
M, N = 8, 10
for r in [1, 2, 3, 5, 10, 20]:
    exp = Experiment(M, N, fp, ratio=r)
    exp.run()









