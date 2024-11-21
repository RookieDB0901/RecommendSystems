import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from collections import namedtuple
from sklearn.preprocessing import  MinMaxScaler, LabelEncoder
import tensorflow as tf
import numpy as np

SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSpareFeat', ['name', 'vocabulary_size', 'embedding_dim', 'max_len'])

# 构造输入层
def get_input_layers(feature_columns):

    # 稠密型特征和稀疏性特征分别创建一个字典，key为特征名，value为一个输入层
    sparse_dict, dense_dict = {}, {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            # 输入训练数据时，会根据Input的name参数匹配数据和输入层
            sparse_dict[fc.name] = Input(shape=(1, ), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_dict[fc.name] = Input(shape=(1, ), name=fc.name)

    return sparse_dict, dense_dict


# 构造稀疏特征的embedding层
def get_embedding_layers(feature_columns, sparse_input_dict, is_linear=False):
    embedding_dict = {}
    # 过滤出稀疏特征
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    # 如果是线性模型，则直接映射成一个标量
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_dict[fc.name] = Embedding(fc.vocabulary_size+1, 1)
    else:
        for fc in sparse_feature_columns:
            embedding_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim)

    return embedding_dict


# 将所有embedding层存入一个列表，以便于concat
def embedding_contact_list(feature_columns, sparse_input_dict, embedding_dict, flatten=False):
    embedding_list = []
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    for fc in sparse_feature_columns:
        _input = sparse_input_dict[fc.name]
        _embedding = embedding_dict[fc.name]
        embed = _embedding(_input)
        # 将嵌入向量展平
        if flatten:
            embed = Flatten()(embed)
        embedding_list.append(embed)

    return embedding_list


def dnn(dnn_input):

    # dnn层，这里的Dropout参数，Dense中的参数都可以自己设定
    fc_layer = Dropout(0.5)(Dense(1024, activation='relu')(dnn_input))
    fc_layer = Dropout(0.3)(Dense(512, activation='relu')(fc_layer))
    dnn_out = Dropout(0.1)(Dense(256, activation='relu')(fc_layer))

    return dnn_out


class CrossNet(Layer):

    def __init__(self, layer_num=3):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num

    def build(self, input_shape):
        # w的维度和输入向量的维度相同
        self.dim = input_shape[-1]

        # 每个交叉层都需要一组系数
        self.W = [self.add_weight(shape=(self.dim,)) for i in range(self.layer_num)]
        self.b = [self.add_weight(shape=(self.dim,)) for i in range(self.layer_num)]

    def call(self, x):

        x_0 = x
        x_l = x_0
        for i in range(self.layer_num):
            # axes=(1, 0): 将x_l的第1个维度与w[i]的第0个维度计算点积
            xl_w = tf.tensordot(x_l, self.W[i], axes=(1, 0))  # xl_w的shape为(batchsize,)
            xl_w = tf.expand_dims(xl_w, axis=-1)  # (batchsize, 1)
            cross = tf.multiply(x_0, xl_w)  # 利用广播机制按位相乘
            x_l = x_l + self.b[i] + cross

        return x_l


def DCN(feature_columns):

    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，方便后续构建模型
    sparse_input_dict, dense_input_dict = get_input_layers(feature_columns)
    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    # 构建embedding层
    embedding_dict = get_embedding_layers(feature_columns, sparse_input_dict)
    # 将稠密型特征concat起来
    concat_dense = Concatenate(axis=1)(list(dense_input_dict.values()))
    # 将稀疏特征的embedding concat起来
    embedding_list = embedding_contact_list(feature_columns, sparse_input_dict, embedding_dict, flatten=True)
    concat_sparse_embed = Concatenate(axis=1)(embedding_list)
    # 将所有特征concat起来
    concat_input = Concatenate(axis=1)([concat_dense, concat_sparse_embed])
    # DNN
    dnn_output = dnn(concat_input)
    # CrossNet
    cross_output = CrossNet()(concat_input)
    # 将DNN和CrossNet的输出concat起来
    concat_output = Concatenate(axis=1)([dnn_output, cross_output])
    # 输出层
    output_layer = Dense(1, activation='sigmoid')(concat_output)

    model = Model(input_layers, output_layer)

    return model


data = pd.read_csv('dataset/criteo/train.csv')
del data['Id']
# 稠密特征和稀疏特征列表
dense_feas = ['I' + str(i) for i in range(1, 14)]
sparse_feas = ['C' + str(i) for i in range(1, 27)]
# 填补缺失值
data[dense_feas] = data[dense_feas].fillna(0.0)
data[sparse_feas] = data[sparse_feas].fillna('-1')
# 稀疏特征编码
lbe = LabelEncoder()
for fea in sparse_feas:
    data[fea] = lbe.fit_transform(data[fea])
# 稠密特征取对数，防止数据范围过大
for fea in dense_feas:
    data[fea] = data[fea].apply(lambda x: np.log(x+1) if x > -1 else -1)
# 使用SparseFeat和DenseFeat存储特征
feature_columns = [SparseFeat(name=fea, vocabulary_size=data[fea].nunique(), embedding_dim=4) for fea in sparse_feas] + [DenseFeat(name=fea, dimension=1) for fea in dense_feas]

my_model = DCN(feature_columns)
my_model.summary()
my_model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])
# 将输入数据转化成字典的形式输入
train_model_input = {name: data[name] for name in dense_feas + sparse_feas}
my_model.fit(train_model_input, data['Label'].values, batch_size=32, epochs=20, validation_split=0.2, )