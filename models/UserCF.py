import math
import random
class Dataset():

    def __init__(self, fp):
        self.data = self.loadData(fp)

    def loadData(self, fp):
        data = []
        for line in open(fp, "r"):
            data.append(tuple(map(int,line.strip().split("::")[:2])))
        return data

    def splitData(self, M, K, seed=1):
        test, train = [], []
        random.seed(seed)
        for user, item in self.data:
            if random.randint(0, M-1) == K:
                test.append((user, item))
            else:
                train.append((user, item))

        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {user: list(items) for user, items in data_dict.items()}
            return data_dict

        return convert_dict(train), convert_dict(test)

class Metrics():

    def __init__(self, train, test, GetRecommendation):
        self.train = train
        self.test = test
        self.GetRecommendaion = GetRecommendation
        self.recs = self.getRec()

    def getRec(self):
        recs = {}
        for user in self.test:
            if user not in recs:
                recs[user] = self.GetRecommendaion(user)
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
            "precision": self.precision(),
            "recall": self.recall()
        }
        print("Metric:", metric)
        return metric

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
        for item in train(user):
            if item not in items:
                items[item] = 0
            items[item] += 1

    def GetRecommendation(user):
        seen_items = set(train[user])
        recs = {k: items[k] for k in item.items() if k not in seen_items}
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
                sum[u][v] += 1
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
        metrics = {"precision": 0, "recall": 0}
        dataset = Dataset(self.fp)
        for i in range(self.M):
            train, test = dataset.splitData(self.M, i)
            metric = self.worker(train, test)
            metrics = {k: metric[k] + metrics[k] for k in metrics}
        metrics = {k: metrics[k]/self.M for k in metrics}
        print('Average Result (M={}, N={}, K={}): {}'.format(self.M, self.N, self.K, metrics))

fp = "./dataset/ml-1m/ratings.dat"
M, K = 8, 10
N = 0 # 为保持一致而设置，随便填一个值
random_exp = Experiment(M, N, K, fp, rt='Random')
random_exp.run()















