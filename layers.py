from turtle import pd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

class FMLayer(keras.layers.Layer):
    """
    FM模型的特征交叉层
    imput_shape: (batch, feature_num, embed_size)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        # 和的平方
        sum_square = tf.square( tf.reduce_sum(inputs, axis=1))
        # 平方的和
        square_sum = tf.reduce_sum( tf.square(inputs), axis=1)
        # output_shape: ( batch, 1)
        output = 0.5 * tf.reduce_sum(sum_square-square_sum, axis=1, keepdims=True)
        return output

class FFMLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        # Inputs shape: (None, feature_num, feature_num, embedding_size)
        # 后期优化方向：利用gather进行张量运算
        f = inputs.shape[1]
        res = 0
        for i in range(f-1):
            for j in range(i+1, f):
                res += tf.multiply(inputs[:,i,j,:] , inputs[:,j,i,:])
        return tf.reduce_sum(res, axis=-1, keepdims=True)

class Cross_Layer(keras.layers.Layer):
    """
    Deep&Cross模型中的 Cross 层
    """
    def __init__(self, layer_num, reg, **kwargs):
        super().__init__(**kwargs)
        # layer_num: cross模块的层数
        self.layer_num = layer_num
        self.reg = reg

    def build(self, input_shape):
        # 矩阵形式存储权重
        self.cross_weight = self.add_weight(
            shape=(self.layer_num, input_shape[1]),
            initializer='random_normal',
            regularizer=self.reg,
            trainable=True)
        # 矩阵形式存储偏移
        self.bias_weight = self.add_weight(
            shape=(self.layer_num, input_shape[1]),
            initializer='random_normal',
            regularizer=self.reg,
            trainable=True)

    def call(self, inputs):
        x0 = inputs
        xi = x0
        for i in tf.range(self.layer_num):
            # x_j = x_0 * ( x_i * w_j) + b_j + x_i
            temp = tf.reduce_sum((xi*self.cross_weight[i]), keepdims=True)
            xi = x0*temp + self.bias_weight[i] + xi

        return xi

class CIN_Layer(keras.layers.Layer):
    def __init__(self, cin_size, reg, **kwargs):
        super().__init__(**kwargs)
        self.cin_size = cin_size
        self.reg = reg

    def build(self, input_shape):
        num_list = [input_shape[-2]] + self.cin_size
        self.w = [
            self.add_weight(
                shape=(num_list[i+1], num_list[i], input_shape[-2], 1),
                initializer='random_normal',
                regularizer=self.reg,
                trainable=True
            )
            for i in tf.range(len(self.cin_size))
        ]

    def call(self, inputs):
        # inputs.shape： (none, m, d)
        x0 = tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)
        res_list = [tf.transpose(x0, [0,1,3,2,4])]
        # 可利用爱因斯坦求和约定进行简化
        # res_list = [x0]
        # for i in range(len(self.cin_size)):
        #     x1 = tf.einsum('bmd,bld,rlm->brd',x0,res_list[i],self.w[i])
        #     res_list.append(x1)
        # return tf.reduce_sum(tf.concat(res_list[1:],axis=1),axis=2)
        for i in range(len(self.cin_size)):
            # x0: (none, 1, 1, m, d)
            # res_list[i] :(none, 1, hleft, 1, d)
            z = tf.multiply(x0, res_list[i]) # shape: (none, 1, hleft, m, d)
            # self.w[i]: (hright, hlegt, m, 1)
            x = tf.multiply(z, self.w[i]) # shape: (none, hright, hleft, m, d)
            x = tf.reduce_sum(x, axis=[2,3], keepdims=True) # shape: (none, hright, 1, 1, d)
            x = tf.transpose(x, [0,2,1,3,4]) # shape: (none, 1, hright, 1, d)
            res_list.append(x)
        out = tf.concat(res_list[1:], axis=2)
        return tf.reduce_sum(out, axis=[1,3,4])

class InnerProductLayer(keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(*kwargs)

    def build(self, input_shape):
        # 保存计算过程所需的下标，利用gather方法提高计算效率
        temp = tf.constant([(i,j) for i in range(input_shape[1]-1) for j in range(i+1, input_shape[1])])
        self.ids_x = temp[:,0]
        self.ids_y = temp[:,1]

    def call(self, inputs):
        p = tf.raw_ops.GatherV2(params=inputs, indices=self.ids_x, axis=1)
        q = tf.raw_ops.GatherV2(params=inputs, indices=self.ids_y, axis=1)
        out = tf.reduce_sum(p*q, axis=-1)
        return out

class OutProductLayer(keras.layers.Layer):
    def __init__(self, reg, **kwargs):
        super().__init__(**kwargs)
        self.reg = reg

    def build(self, input_shape):
        fields = input_shape[1]
        embed_size = input_shape[-1]
        pair_num = fields*(fields-1)//2
        # 保存计算过程中所需要的下标，利用gather方法提高计算效率
        temp = tf.constant([(i,j) for i in range(fields-1) for j in range(i+1, fields)])
        self.ids_x = temp[:,0]
        self.ids_y = temp[:,1]

        self.w = self.add_weight(
            shape=(pair_num, embed_size, embed_size),
            initializer='random_normal',
            regularizer=self.reg,
            trainable=True,
        )

    def call(self, inputs):
        # 后续可利用爱因斯坦求和约定进行改进 tf.einsum()
        p = tf.raw_ops.GatherV2(params=inputs, indices=self.ids_x, axis=1)
        p = p[...,tf.newaxis]

        q = tf.raw_ops.GatherV2(params=inputs, indices=self.ids_y, axis=1)
        q = tf.expand_dims(q, axis=2)

        out = tf.reduce_sum(p*q*self.w, axis=[-2, -1])
        return out