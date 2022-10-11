import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
预处理函数：
    根据输入的dataframe和settings输出inputs层及预处理层

对于数值类型的输入，拼接后直接输出
shape:(batch, n)

对于类别行输入, 输出每个feature的weight以及embedding。
weight 结果用做逻辑回归层的输入
shape:(batch, 1)
设计目标： 避免类别特征 onehot编码后引起的维数灾难

embedding 的结果可用于两个部分：一是用于深度神经网络层，二是用于特征组合层
shape:(batch, cat_feature_numb, embed_size)
"""

def stack_dict(inputs, fun=tf.stack):
    # 将格式统一转化为tf.float32之后拼接在一起
    value = []
    for key in inputs.keys():
        value.append(tf.cast(inputs[key], tf.float32))

    return fun(value,axis=-1)

def process(df, settings):
    #生成输入层
    inputs = {}
    dense_inputs = {}
    for name, column in df.items():
        if type(column[0]) == str:
            dtype = tf.string
        elif type(column[0]) == int:
            dtype = tf.int64
        else:
            dtype = tf.float32
        inputs[name] = keras.Input(shape=(), name=name, dtype=dtype)
        if name in settings.dense_feature:
            dense_inputs[name] = inputs[name]
    # 拼接数值特征
    dense_inputs = stack_dict(dense_inputs)
    
    # 生成类别特征的 weight 和 embedding
    cat_weight, cat_embed = createCatEmbed(
        df=df,
        inputs=inputs,
        settings=settings
    )
    
    return inputs, (dense_inputs, cat_weight, cat_embed)

def createCatEmbed(df, inputs, settings):
    """生成类别特征的 weight 和 embedding"""
    # Create OneHot and Embedding for Category Feature
    cat_weight = []
    cat_embed = []
    feature_size = len(settings.cat_feature)
    for name in settings.cat_feature:
        # 将类别特征转化为自然数表示
        vocabulary = list(set(df[name]))
        cat_size = len(vocabulary)
        lookup_int = keras.layers.StringLookup(vocabulary=vocabulary, output_mode='int', name=name+'_int')
        # 利用embedding函数生成weight和embedding
        weight = keras.layers.Embedding(
            input_dim = cat_size + 1,
            output_dim = 1,
            embeddings_regularizer = settings.reg,
            name = name + '_weight'
        )
        if settings.embed_mode == 'vector':
            embed = keras.layers.Embedding(input_dim=cat_size+1, 
                                           output_dim=settings.embed_size,
                                           embeddings_regularizer=settings.reg,
                                           name=name+'_embed',)
        elif settings.embed_mode == 'matrix':
            embed = CatEmbed2D(shape=(cat_size+1,feature_size,settings.embed_size),
                                    reg=settings.reg,
                                    name=name+'_matrixEmbed',) 
        x = lookup_int(inputs[name])
        x_1 = weight(x)
        x_2 = embed(x)
        cat_weight.append(x_1)
        cat_embed.append(x_2)

    cat_weight = keras.layers.Add()(cat_weight)
    cat_embed = tf.stack(cat_embed, axis=1)
    # output shape { weight:( batch, 1), embed:( batch, feature_numb, embed_shape)}
    return cat_weight, cat_embed

class CatEmbed2D(keras.layers.Layer):
    # 为FFM模型生成类别特征的矩阵 embedding
    def __init__(self, shape, reg, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.reg = reg

    def build(self, input_shape):
        self.embed_weight = self.add_weight(
            shape=self.shape,
            initializer='random_normal',
            regularizer=self.reg,
            trainable=True
        )

    def call(self, inputs):
        return tf.gather(self.embed_weight, inputs)