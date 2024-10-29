import pandas as pd
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchkeras import summary
from sklearn.metrics import roc_curve, roc_auc_score, auc
import datetime
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/processed_train.csv')
test = pd.read_csv('data/processed_test.csv')
val = pd.read_csv('data/processed_val.csv')
print(train.head())
print(test.head())
print(val.head())

data = pd.concat((train, val, test))
print(data.head())

dense_feas = ['I' + str(i) for i in range(1, 14)]
sparse_feas = ['C' + str(i) for i in range(1, 27)]
sparse_val_nums = {}
for fea in sparse_feas:
    sparse_val_nums[fea] = data[fea].nunique()
feature_info = [dense_feas, sparse_feas, sparse_val_nums]

td_train = TensorDataset(torch.tensor(train.drop(columns='Label').values).float(), torch.tensor(train['Label'].values).float())
td_val = TensorDataset(torch.tensor(val.drop(columns='Label').values).float(), torch.tensor(val['Label'].values).float())

train_set = DataLoader(td_train, shuffle=True, batch_size=16)
val_set = DataLoader(td_val, shuffle=True, batch_size=16)

class ResBlock(nn.Module):

    def __init__(self, hidden, n_dim):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(n_dim, hidden)
        self.linear2 = nn.Linear(hidden, n_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.linear2(self.relu(self.linear1(x)))
        return self.relu(res + x)

class DeepCrossing(nn.Module):

    def __init__(self, feature_info, hiddens, embed_dim=10, dropout=0.1, dim_output=1):
        """
        :param feature_info: [dense_feas, sparse_feas, sparse_val_nums]
        :param hiddens: 每个残差块的隐藏层维度组成的列表
        :param embed_dim: 每个离散特征embedding后的维度
        :param dropout: dropout概率
        :param dim_output: 输出维度
        """
        super(DeepCrossing, self).__init__()
        self.dense_feas, self.sparse_feas, self.sparse_val_nums = feature_info
        self.embeddings = nn.ModuleDict({"embed_" + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embed_dim) for key, val in self.sparse_val_nums.items()})
        embed_dim_sum = sum([embed_dim] * len(self.sparse_feas))
        dim_stack = embed_dim_sum + len(self.dense_feas)
        self.res_layers = nn.ModuleList([ResBlock(hidden, dim_stack) for hidden in hiddens])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim_stack, dim_output)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()  # 需要转成长张量， 这个是embedding的输入要求格式
        sparse_embeds = [self.embeddings["embed_"+key](sparse_inputs[:, i]) for key, i in zip(self.sparse_val_nums.keys(), range(sparse_inputs.shape[1]))]
        sparse_embed = torch.cat(sparse_embeds, axis=-1)
        stack = torch.cat([sparse_embed, dense_inputs], axis=-1)
        # stack = torch.cat([torch.cat(sparse_embeds, axis=-1), dense_inputs], axis=-1)
        for res_layer in self.res_layers:
            stack = res_layer(stack)
        output = self.linear(self.dropout(stack))
        output = F.sigmoid(output)
        return output

hiddens = [64, 128, 256, 128, 64, 32]
net = DeepCrossing(feature_info, hiddens)
summary(net, input_shape=(train.shape[1],))

for fea, label in iter(train_set):
    out = net(fea)
    print(out)
    break


def auc(y_pred, y_true):
    return roc_auc_score(y_true.data, y_pred.data)


loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
metric_func = auc
metric_name = "auc"

epochs = 10
log_step_freq = 10

dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
print('Start Training...')
nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('=========' * 8 + "%s" % nowtime)

for epoch in range(1, epochs + 1):
    # 训练阶段
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1

    for step, (features, labels) in enumerate(train_set, 1):

        # 梯度清零
        optimizer.zero_grad()

        # 正向传播
        predictions = net(features)
        loss = loss_func(predictions, labels.unsqueeze(1))
        try:  # 这里就是如果当前批次里面的y只有一个类别， 跳过去
            metric = metric_func(predictions, labels.unsqueeze(1))
            metric_sum += metric.item()
        except ValueError:
            pass

        # 反向传播求梯度
        loss.backward()
        optimizer.step()

        # 打印batch级别日志
        loss_sum += loss.item()
        if step % log_step_freq == 0:
            print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                  (step, loss_sum / step, metric_sum / step))

    # 验证阶段
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features, labels) in enumerate(val_set, 1):
        with torch.no_grad():
            predictions = net(features)
            val_loss = loss_func(predictions, labels.unsqueeze(1))
            try:
                val_metric = metric_func(predictions, labels.unsqueeze(1))
                val_metric_sum += val_metric.item()
            except ValueError:
                pass
        val_loss_sum += val_loss.item()

    # 记录日志
    info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
    dfhistory.loc[epoch - 1] = info

    # 打印epoch级别日志
    print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
           "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
          % info)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)

print('Finished Training...')



