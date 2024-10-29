import math
import random
from utils import Dataset, Metrics


# 随机推荐
def Random(train, N, K):
    items = {}
    for user in train:
        for item in train[user]:
                items[item] = 1

    def GetRecommendation(user):
        seen_items = set(train[user])
        recs = {k: items[k] for k in items if k not in seen_items}
        recs = list(recs.items())
        random.shuffle(recs)
        return recs[:K]

    return GetRecommendation

def MostPopular(train, N, K):
    items = {}
    for user in train:
        for item in train[user]:
            if item not in items:
                items[item] = 0
            items[item] += 1

    def GetRecommendation(user):
        seen_items = set(train[user])
        recs = {k: items[k] for k in items if k not in seen_items}
        recs = list(sorted(recs.items(), key=lambda x: x[1], reverse=True))
        return recs[:K]

    return GetRecommendation


def UserCF(train, N, K):
    """
    找到N个相似用户，推荐相似用户的历史物品
    :param train:
    :param N:
    :param K:
    :return:
    """
    user_items = {}
    for user in train:
        for item in train[user]:
            if item not in user_items:
                user_items[item] = []
            user_items[item].append(user)
    sim = {}
    num = {}
    for item in user_items:
        users = user_items[item]
        for i in range(len(users)):
            u = users[i]
            if u not in num:
                num[u]=0
            num[u] += 1
            if u not in sim:
                sim[u] = {}
            for j in range(len(users)):
                if i ==j:
                    continue
                v = users[j]
                if v not in sim[u]:
                    sim[u][v] = 0
                sim[u][v] += 1
    for u in sim:
        for v in sim[u]:
            sim[u][v] /= math.sqrt(num[u] * num[v])

    sorted_sim = {k: list(sorted(v.items(), key=lambda x: x[1], reverse=True)) for k, v in sim.items()}

    def GetRecommendatiion(user):
        seen_items = set(train[user])
        items = {}
        recs = []
        for u, _ in sorted_sim[user][:N]:
            for item in train[u]:
                if item not in seen_items:
                    if item not in items:
                        items[item] = 0
                    items[item] += sim[user][u]
        recs = list(sorted(items.items(), key = lambda x: x[1], reverse = True))[:K]
        return recs
    return GetRecommendatiion


class Experiment():

    def __init__(self, M, N, K, fp, rt = "UserCF"):
        self.M = M
        self.N = N
        self.K = K
        self.fp = fp
        self.rt = rt
        self.alg = {"UserCF": UserCF, "Random": Random, "MostPopular": MostPopular}

    def worker(self, train, test):
        GetRecommendation = self.alg[self.rt](train, self.N, self.K)
        metric = Metrics(train, test, GetRecommendation)
        return metric.eval()

    def run(self):
        metrics = {"Precision": 0, "Recall": 0}
        dataset = Dataset(self.fp)
        for i in range(self.M):
            train, test = dataset.splitData(self.M, i)
            metric = self.worker(train, test)
            metrics = {k: metric[k] + metrics[k] for k in metrics}
        metrics = {k: metrics[k]/self.M for k in metrics}
        print('Average Result (M={}, N={}, K={}): {}'.format(self.M, self.N, self.K, metrics))

fp = "./dataset/ml-1m/ratings.dat"

# 1. Random实验
print("Random:")
M, K = 8, 10
N = 0 # 为保持一致而设置，随便填一个值
random_exp = Experiment(M, N, K, fp, rt='Random')
random_exp.run()

# 2. MostPopular实验
print("MostPopular:")
M, N = 8, 10
K = 0 # 为保持一致而设置，随便填一个值
mp_exp = Experiment(M, K, N, fp, rt='MostPopular')
mp_exp.run()

# 3. UserCF实验
print("UserFC:")
M, N = 8, 10
for K in [5, 10, 20, 40, 80, 160]:
    cf_exp = Experiment(M, K, N, fp, rt='UserCF')
    cf_exp.run()












