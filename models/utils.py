import random

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


class Metrics():
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

